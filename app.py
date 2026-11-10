from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for, make_response
import cv2
import numpy as np
import re
import sqlite3
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import time
import csv
from io import StringIO

app = Flask(__name__)
app.secret_key = 'super_secret_parking_key'

TOTAL_SPACES = 200


# ================= 1. 数据库初始化 =================
def init_db():
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS current_vehicles (
                        plate_number TEXT PRIMARY KEY, entry_time TEXT NOT NULL, is_vip INTEGER DEFAULT 0)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS parking_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, plate_number TEXT NOT NULL, 
                        entry_time TEXT NOT NULL, exit_time TEXT NOT NULL, 
                        duration_hours REAL NOT NULL, fee REAL NOT NULL, phone TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS whitelist (
                        plate_number TEXT PRIMARY KEY, owner_name TEXT NOT NULL, expire_date TEXT NOT NULL,
                        payment_history TEXT NOT NULL, status INTEGER DEFAULT 0,
                        phone TEXT, building TEXT, unit TEXT, room TEXT)''')
    try:
        cursor.execute("ALTER TABLE parking_records ADD COLUMN phone TEXT")
    except:
        pass
    cols = [("status", "INTEGER DEFAULT 0"), ("phone", "TEXT"), ("building", "TEXT"), ("unit", "TEXT"),
            ("room", "TEXT")]
    for col, dtype in cols:
        try:
            cursor.execute(f"ALTER TABLE whitelist ADD COLUMN {col} {dtype}")
        except:
            pass
    conn.commit()
    return conn


conn = init_db()

# ================= 2. 模型加载 =================
print("正在加载 AI 模型...")
yolo_model = YOLO('best.pt')
ocr = PaddleOCR(lang="ch", enable_mkldnn=False)
print("模型加载完毕！")

cap_main = cv2.VideoCapture(1)  # 这里如果是你的 iVCam 手机摄像头，改成 1
active_plates = {}
DISAPPEAR_TIMEOUT = 5
# 全局状态：增加 need_phone 用于触发前端弹窗
latest_log = {"plate": "等待识别", "status": "门禁监控中", "time": "", "type": "normal", "need_phone": False,
              "temp_plate": "", "expire_date": ""}
last_trigger_time = {}


def put_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        fontStyle = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", textSize, encoding="utf-8")
    except:
        fontStyle = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def calculate_fee(duration_hours):
    if duration_hours <= 1: return 5.0
    return 5.0 + (duration_hours - 1) * 3.0


# ================= 3. 核心业务逻辑 =================
def process_vehicle(plate_text):
    global latest_log
    cursor = conn.cursor()

    plate_regex = r'^[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]{1}[A-Z]{1}[A-Z0-9]{5,6}$'
    if not re.match(plate_regex, plate_text):
        latest_log = {"plate": plate_text, "status": "非合法车牌，拒绝录入", "time": datetime.now().strftime("%H:%M:%S"),
                      "type": "warning", "need_phone": False, "temp_plate": "", "expire_date": ""}
        return

    cursor.execute("SELECT entry_time, is_vip FROM current_vehicles WHERE plate_number = ?", (plate_text,))
    result = cursor.fetchone()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if not result:
        # 【入场逻辑】
        cursor.execute("SELECT expire_date, status FROM whitelist WHERE plate_number = ?", (plate_text,))
        whitelist_result = cursor.fetchone()

        if whitelist_result:
            expire_date, vip_status = whitelist_result
            now_date = datetime.now().strftime("%Y-%m-%d")
            if expire_date >= now_date and vip_status == 0:
                # 业主直接放行入库
                cursor.execute("INSERT INTO current_vehicles (plate_number, entry_time, is_vip) VALUES (?, ?, 1)",
                               (plate_text, now_str))
                conn.commit()
                latest_log = {"plate": plate_text, "status": "业主月租车，直接放行", "time": now.strftime("%H:%M:%S"),
                              "type": "vip", "need_phone": False, "temp_plate": "", "expire_date": expire_date}
                return
            else:
                latest_log = {"plate": plate_text, "status": "业主欠费/到期，请缴费通行",
                              "time": now.strftime("%H:%M:%S"), "type": "warning", "need_phone": False,
                              "temp_plate": "", "expire_date": expire_date}
                return

        cursor.execute("SELECT COUNT(*) FROM current_vehicles")
        if cursor.fetchone()[0] >= TOTAL_SPACES:
            latest_log = {"plate": plate_text, "status": "车位已满，禁止入场", "time": now.strftime("%H:%M:%S"),
                          "type": "warning", "need_phone": False, "temp_plate": "", "expire_date": ""}
            return

        # ★ 关键修改：临停车辆不直接入库，先触发前端弹窗
        latest_log = {"plate": plate_text, "status": "临停车辆，请扫码登记手机号", "time": now.strftime("%H:%M:%S"),
                      "type": "entry", "need_phone": True, "temp_plate": plate_text, "expire_date": ""}
        return

    else:
        # 【出场逻辑】
        entry_time = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
        is_vip = result[1]

        # 10秒内重复识别不处理
        if (now - entry_time).total_seconds() < 10:
            latest_log = {"plate": plate_text, "status": "车辆已入场，请勿重复识别", "time": now.strftime("%H:%M:%S"),
                          "type": "normal", "need_phone": False, "temp_plate": "", "expire_date": ""}
            return

        if not is_vip:
            duration_hours = max(1, round((now - entry_time).total_seconds() / 3600))
            fee = calculate_fee(duration_hours)

            # 从数据库取这辆车的手机号（入场时已经写入了）
            cursor.execute("SELECT phone FROM parking_records WHERE plate_number = ? ORDER BY id DESC LIMIT 1",
                           (plate_text,))
            phone_row = cursor.fetchone()
            actual_phone = phone_row[0] if phone_row else "未登记"

            cursor.execute('''INSERT INTO parking_records (plate_number, entry_time, exit_time, duration_hours, fee, phone)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (plate_text, result[0], now_str, duration_hours, fee, actual_phone))
            latest_log = {"plate": plate_text, "status": f"出场成功，时长{duration_hours}小时，收费{fee}元",
                          "time": now.strftime("%H:%M:%S"), "type": "exit", "need_phone": False, "temp_plate": "",
                          "expire_date": ""}
        else:
            latest_log = {"plate": plate_text, "status": "业主月租车，出场放行", "time": now.strftime("%H:%M:%S"),
                          "type": "vip", "need_phone": False, "temp_plate": "", "expire_date": ""}

        cursor.execute("DELETE FROM current_vehicles WHERE plate_number = ?", (plate_text,))
        conn.commit()


def generate_frames():
    global active_plates
    while True:
        success, frame = cap_main.read()
        if not success:
            time.sleep(0.1)
            continue
        now_ts = time.time()
        keys_to_remove = [k for k, v in active_plates.items() if now_ts - v > DISAPPEAR_TIMEOUT]
        for k in keys_to_remove: del active_plates[k]

        if int(now_ts * 10) % 3 == 0:
            results = yolo_model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pad_x, pad_y = int((x2 - x1) * 0.08), int((y2 - y1) * 0.15)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(frame.shape[1], x2 + pad_x), min(frame.shape[0], y2 + pad_y)

                    plate_img = frame[cy1:cy2, cx1:cx2]
                    if plate_img.size == 0: continue

                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    if cv2.Laplacian(gray, cv2.CV_64F).var() < 30: continue

                    plate_img_large = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    ocr_result = ocr.predict(plate_img_large)

                    if ocr_result and 'rec_texts' in ocr_result[0]:
                        res = ocr_result[0]
                        text_blocks = []
                        if 'rec_polys' in res:
                            for text, poly in zip(res['rec_texts'], res['rec_polys']):
                                min_x = int(min(p[0] for p in poly))
                                text_blocks.append((min_x, text))
                            text_blocks.sort(key=lambda x: x[0])
                            raw_text = "".join([t[1] for t in text_blocks])
                        else:
                            raw_text = "".join(res['rec_texts'])

                        plate_text = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9]', '', raw_text)

                        if len(plate_text) >= 7:
                            active_plates[plate_text] = now_ts
                            if plate_text not in last_trigger_time or (now_ts - last_trigger_time[plate_text]) > 1.5:
                                last_trigger_time[plate_text] = now_ts
                                process_vehicle(plate_text)

                            color = (0, 255, 0)
                            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 2)
                            frame = put_chinese_text(frame, plate_text, (cx1, cy1 - 35), textColor=color)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ================= 4. 路由 API =================
@app.route('/')
def index(): return render_template('index.html')


@app.route('/api/log')
def get_log(): return jsonify(latest_log)


@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ★ 关键修改：接收手机号后，真正执行入场写库操作
@app.route('/api/register_phone', methods=['POST'])
def register_phone():
    data = request.json
    plate = data.get('plate')
    phone = data.get('phone')

    if plate and phone:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = conn.cursor()
        try:
            # 真正写入在场表
            cursor.execute("INSERT INTO current_vehicles (plate_number, entry_time, is_vip) VALUES (?, ?, 0)",
                           (plate, now_str))
            # 预插入一条流水记录，把手机号存进去，出场时直接更新出场时间和费用
            cursor.execute('''INSERT INTO parking_records (plate_number, entry_time, exit_time, duration_hours, fee, phone)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (plate, now_str, "--:--:--", 0, 0, phone))
            conn.commit()

            global latest_log
            latest_log = {"plate": plate, "status": "登记成功，闸机已开启", "time": datetime.now().strftime("%H:%M:%S"),
                          "type": "entry", "need_phone": False, "temp_plate": "", "expire_date": ""}
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "msg": str(e)}), 500
    return jsonify({"success": False}), 400


# 取消手机号登记接口（点了 X 关闭弹窗时调用，清理状态，防止车进不去）
@app.route('/api/cancel_phone', methods=['POST'])
def cancel_phone():
    global latest_log
    latest_log = {"plate": "等待识别", "status": "门禁监控中", "time": datetime.now().strftime("%H:%M:%S"),
                  "type": "normal", "need_phone": False, "temp_plate": "", "expire_date": ""}
    return jsonify({"success": True})


@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM current_vehicles WHERE is_vip = 0")
    temp_current = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM current_vehicles WHERE is_vip = 1")
    vip_current = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(fee) FROM parking_records WHERE exit_time != '--:--:--'")
    total_income = cursor.fetchone()[0] or 0.0
    cursor.execute("SELECT COUNT(*) FROM whitelist")
    whitelist_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT substr(exit_time, 1, 10) as date, SUM(fee) as revenue FROM parking_records WHERE exit_time >= date('now', '-7 days') AND exit_time != '--:--:--' GROUP BY date ORDER BY date")
    trend_data = cursor.fetchall()
    cursor.execute(
        "SELECT substr(exit_time, 12, 2) as hour, COUNT(*) as count FROM parking_records WHERE exit_time != '--:--:--' GROUP BY hour ORDER BY hour")
    hour_data = cursor.fetchall()
    cursor.execute(
        "SELECT duration_hours, COUNT(*) FROM parking_records WHERE exit_time != '--:--:--' GROUP BY duration_hours")
    dur_data = cursor.fetchall()
    cursor.execute(
        "SELECT plate_number, exit_time, fee FROM parking_records WHERE exit_time != '--:--:--' ORDER BY id DESC LIMIT 5")
    recent_records = cursor.fetchall()

    return jsonify({
        "temp_current": temp_current, "vip_current": vip_current,
        "total_income": round(total_income, 2), "whitelist_count": whitelist_count,
        "hours": [f"{r[0]}点" for r in hour_data], "hour_counts": [r[1] for r in hour_data],
        "dur_values": [{"name": f"{r[0]}小时", "value": r[1]} for r in dur_data],
        "recent_records": [{"plate": r[0], "time": r[1], "fee": r[2]} for r in recent_records],
        "trend_dates": [r[0] for r in trend_data], "trend_incomes": [r[1] for r in trend_data],
        "total_spaces": TOTAL_SPACES
    })


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == 'admin123':
            session['is_admin'] = True
            return redirect(url_for('admin'))
        return render_template('login.html', error="密码错误")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('login'))


@app.route('/admin')
def admin():
    if not session.get('is_admin'): return redirect(url_for('login'))
    return render_template('admin.html')


@app.route('/api/whitelist', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_whitelist():
    if not session.get('is_admin'): return jsonify({"error": "未授权"}), 401
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()
    if request.method == 'GET':
        page = int(request.args.get('page', 1))
        search = request.args.get('search', '')
        limit, offset = 7, (page - 1) * 7
        query = "SELECT plate_number, owner_name, expire_date, payment_history, phone, building, unit, room FROM whitelist"
        count_query = "SELECT COUNT(*) FROM whitelist"
        params = []
        if search:
            query += " WHERE plate_number LIKE ? OR owner_name LIKE ? OR building LIKE ?"
            count_query += " WHERE plate_number LIKE ? OR owner_name LIKE ? OR building LIKE ?"
            params = [f"%{search}%", f"%{search}%", f"%{search}%"]
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        query += " LIMIT ? OFFSET ?"
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()
        data = [{"plate": r[0], "owner": r[1], "expire": r[2], "history": r[3], "phone": r[4] or "未填写",
                 "building": r[5] or "未填写", "unit": r[6] or "未填写", "room": r[7] or "未填写"} for r in rows]
        return jsonify({"data": data, "total": total})
    elif request.method == 'POST':
        data = request.json
        try:
            cursor.execute(
                """INSERT INTO whitelist (plate_number, owner_name, expire_date, payment_history, phone, building, unit, room, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (data['plate'], data['owner'], data['expire'], data['history'], data['phone'], data['building'],
                 data['unit'], data['room'], data.get('status', 0)))
            conn.commit()
            return jsonify({"success": True})
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "msg": "该车牌号已存在"}), 400
    elif request.method == 'PUT':
        data = request.json
        try:
            cursor.execute(
                """UPDATE whitelist SET owner_name=?, expire_date=?, payment_history=?, phone=?, building=?, unit=?, room=?, status=? WHERE plate_number=?""",
                (data['owner'], data['expire'], data['history'], data['phone'], data['building'], data['unit'],
                 data['room'], data.get('status', 0), data['plate']))
            conn.commit()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"success": False, "msg": str(e)}), 400
    elif request.method == 'DELETE':
        plate = request.args.get('plate')
        if plate:
            cursor.execute("DELETE FROM whitelist WHERE plate_number = ?", (plate,))
            conn.commit()
            return jsonify({"success": True})
        return jsonify({"success": False, "msg": "参数缺失"}), 400


@app.route('/api/records')
def get_records():
    if not session.get('is_admin'): return jsonify({"error": "未授权"}), 401
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()
    page = int(request.args.get('page', 1))
    search_plate = request.args.get('plate', '')
    search_phone = request.args.get('phone', '')
    limit, offset = 7, (page - 1) * 7

    base_query = """
        SELECT id, plate_number, entry_time, exit_time, duration_hours, fee, phone,
               CASE WHEN exit_time = '--:--:--' THEN 1 ELSE 0 END as is_inside
        FROM parking_records
        WHERE 1=1
    """
    count_query = "SELECT COUNT(*) FROM parking_records WHERE 1=1"
    params = []

    if search_plate:
        base_query += " AND plate_number LIKE ?"
        count_query += " AND plate_number LIKE ?"
        params.append(f"%{search_plate}%")
    if search_phone:
        base_query += " AND phone LIKE ?"
        count_query += " AND phone LIKE ?"
        params.append(f"%{search_phone}%")

    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]

    final_query = base_query + " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
    cursor.execute(final_query, params + [limit, offset])
    rows = cursor.fetchall()

    data = [{
        "id": r[0], "plate": r[1], "entry": r[2], "exit": r[3],
        "duration": r[4], "fee": r[5], "phone": r[6] or "未登记", "is_inside": r[7]
    } for r in rows]

    return jsonify({"data": data, "total": total})


@app.route('/api/export')
def export_records():
    if not session.get('is_admin'): return "未授权", 401
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()
    # 只导出真正出场的流水
    cursor.execute(
        "SELECT id, plate_number, entry_time, exit_time, duration_hours, fee, phone FROM parking_records WHERE exit_time != '--:--:--'")
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['记录ID', '车牌号', '入场时间', '出场时间', '停车时长(小时)', '收费(元)', '手机号'])
    cw.writerows(cursor.fetchall())
    output = make_response(si.getvalue().encode('utf-8-sig'))
    output.headers["Content-Disposition"] = "attachment; filename=parking_records.csv"
    output.headers["Content-type"] = "text/csv"
    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)