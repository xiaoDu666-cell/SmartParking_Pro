from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for,make_response
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
app.secret_key = 'super_secret_parking_key'  # 用于 session 加密



# ================= 1. 数据库初始化 =================
def init_db():
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS current_vehicles (
                        plate_number TEXT PRIMARY KEY, entry_time TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS parking_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, plate_number TEXT NOT NULL, 
                        entry_time TEXT NOT NULL, exit_time TEXT NOT NULL, 
                        duration_hours REAL NOT NULL, fee REAL NOT NULL)''')

    # 新增 status 字段 (0: 已缴费/正常, 1: 欠费)
    cursor.execute('''CREATE TABLE IF NOT EXISTS whitelist (
                        plate_number TEXT PRIMARY KEY, 
                        owner_name TEXT NOT NULL,
                        expire_date TEXT NOT NULL,
                        payment_history TEXT NOT NULL,
                        status INTEGER DEFAULT 0)''')

    # 防止旧数据库没这个字段导致报错
    try:
        cursor.execute("ALTER TABLE whitelist ADD COLUMN status INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # 字段已存在

    conn.commit()
    return conn

conn = init_db()

# ================= 2. 模型加载 =================
print("正在加载 AI 模型...")
yolo_model = YOLO('best.pt')
ocr = PaddleOCR(lang="ch", enable_mkldnn=False)
print("模型加载完毕！")

# ================= 3. 单通道视频源 =================
cap_main = cv2.VideoCapture(1)  # 假设只有一个门禁摄像头

active_plates = {}
DISAPPEAR_TIMEOUT = 5
latest_log = {"plate": "等待识别", "status": "门禁监控中", "time": "", "type": "normal"}
last_trigger_time = {}


def put_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        fontStyle = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", textSize, encoding="utf-8")
    except:
        fontStyle = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ================= 4. 核心业务逻辑 =================
def calculate_fee(duration_hours):
    if duration_hours <= 1: return 5.0
    return 5.0 + (duration_hours - 1) * 3.0


# 全局车位总数（设置个大一点的车位）
TOTAL_SPACES = 300


def process_vehicle(plate_text):
    global latest_log
    cursor = conn.cursor()

    # ★ 修复问题 4：严格的国内车牌正则校验（防止乱码入库）
    # 匹配规则：省份汉字 + 字母 + 5位字母数字（包括新能源8位）
    plate_regex = r'^[京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新]{1}[A-Z]{1}[A-Z0-9]{5,6}$'
    if not re.match(plate_regex, plate_text):
        latest_log = {"plate": plate_text, "status": "非合法车牌，拒绝录入", "time": datetime.now().strftime("%H:%M:%S"),
                      "type": "warning"}
        return  # 直接终止，不写库

    # 1. 判定是否为白名单业主
    cursor.execute("SELECT expire_date, status FROM whitelist WHERE plate_number = ?", (plate_text,))
    whitelist_result = cursor.fetchone()

    if whitelist_result:
        now_date = datetime.now().strftime("%Y-%m-%d")
        expire_date = whitelist_result[0]
        is_owe = whitelist_result[1] == 1

        if expire_date >= now_date and not is_owe:
            latest_log = {"plate": plate_text, "status": "业主月租车，直接放行",
                          "time": datetime.now().strftime("%H:%M:%S"), "type": "vip"}
            return
        else:
            latest_log = {"plate": plate_text, "status": "业主欠费或已到期，请缴费通行",
                          "time": datetime.now().strftime("%H:%M:%S"), "type": "warning"}
            return  # ★ 修复问题 3：欠费直接拒绝放行，不再走临停逻辑

    # 2. 临停车辆处理
    cursor.execute("SELECT entry_time FROM current_vehicles WHERE plate_number = ?", (plate_text,))
    result = cursor.fetchone()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if result:
        # 出场
        entry_time = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
        duration_hours = max(1, round((now - entry_time).total_seconds() / 3600))
        fee = calculate_fee(duration_hours)
        cursor.execute("DELETE FROM current_vehicles WHERE plate_number = ?", (plate_text,))
        cursor.execute('''INSERT INTO parking_records (plate_number, entry_time, exit_time, duration_hours, fee)
                          VALUES (?, ?, ?, ?, ?)''', (plate_text, result[0], now_str, duration_hours, fee))
        conn.commit()
        latest_log = {"plate": plate_text, "status": f"临停出场，收费 {fee} 元", "time": now.strftime("%H:%M:%S"),
                      "type": "exit"}
    else:
        # 入场（检查车位是否已满）
        cursor.execute("SELECT COUNT(*) FROM current_vehicles")
        current_count = cursor.fetchone()[0]
        if current_count >= TOTAL_SPACES:
            latest_log = {"plate": plate_text, "status": "车位已满，禁止入场", "time": now.strftime("%H:%M:%S"),
                          "type": "warning"}
            return

        cursor.execute("INSERT INTO current_vehicles (plate_number, entry_time) VALUES (?, ?)", (plate_text, now_str))
        conn.commit()
        latest_log = {"plate": plate_text, "status": "临时车辆入场", "time": now.strftime("%H:%M:%S"), "type": "entry"}

def generate_frames():
    global active_plates
    while True:
        success, frame = cap_main.read()
        if not success:
            time.sleep(0.1);
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


# ================= 5. 路由与 API =================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/log')
def get_log():
    return jsonify(latest_log)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()

    # 1. 核心指标 (针对小区：临停收费、场内临停、白名单总数)
    cursor.execute("SELECT COUNT(*) FROM current_vehicles")
    current_count = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(fee) FROM parking_records")
    total_income = cursor.fetchone()[0] or 0.0
    cursor.execute("SELECT COUNT(*) FROM whitelist")
    whitelist_count = cursor.fetchone()[0]

    # 2. 24小时临停车流高峰分析
    cursor.execute("""
        SELECT substr(exit_time, 12, 2) as hour, COUNT(*) as count 
        FROM parking_records GROUP BY hour ORDER BY hour
    """)
    hour_data = cursor.fetchall()
    hours = [f"{row[0]}点" for row in hour_data]
    hour_counts = [row[1] for row in hour_data]

    # 3. 停车时长占比分析
    cursor.execute("SELECT duration_hours, COUNT(*) FROM parking_records GROUP BY duration_hours")
    dur_data = cursor.fetchall()
    dur_values = [{"name": f"{row[0]}小时", "value": row[1]} for row in dur_data]

    # 4. 最新流水前5
    cursor.execute("SELECT plate_number, exit_time, fee FROM parking_records ORDER BY id DESC LIMIT 5")
    recent_records = [{"plate": row[0], "time": row[1], "fee": row[2]} for row in cursor.fetchall()]

    return jsonify({
        "current_count": current_count, "total_income": round(total_income, 2), "whitelist_count": whitelist_count,
        "hours": hours, "hour_counts": hour_counts, "dur_values": dur_values, "recent_records": recent_records
    })


# --- 后台登录鉴权 ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == 'admin123':
            session['is_admin'] = True
            return redirect(url_for('admin'))
        return render_template('login.html', error="密码错误，请重试")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('login'))


# --- 受保护的后台管理页面 ---
@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    return render_template('admin.html')


# --- 白名单 API (受保护) ---
@app.route('/api/whitelist', methods=['GET', 'POST'])
def manage_whitelist():
    if not session.get('is_admin'):
        return jsonify({"error": "未授权"}), 401

    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()

    if request.method == 'GET':
        page = int(request.args.get('page', 1))
        search = request.args.get('search', '')
        limit = 8
        offset = (page - 1) * limit

        query = "SELECT * FROM whitelist"
        count_query = "SELECT COUNT(*) FROM whitelist"
        params = []

        if search:
            query += " WHERE plate_number LIKE ? OR owner_name LIKE ?"
            count_query += " WHERE plate_number LIKE ? OR owner_name LIKE ?"
            params = [f"%{search}%", f"%{search}%"]

        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]

        query += " LIMIT ? OFFSET ?"
        cursor.execute(query, params + [limit, offset])
        rows = cursor.fetchall()

        data = [{"plate": r[0], "owner": r[1], "expire": r[2], "history": r[3]} for r in rows]
        return jsonify({"data": data, "total": total})

    elif request.method == 'POST':
        data = request.json
        try:
            cursor.execute(
                "INSERT INTO whitelist (plate_number, owner_name, expire_date, payment_history) VALUES (?, ?, ?, ?)",
                (data['plate'], data['owner'], data['expire'], data['history']))
            conn.commit()
            return jsonify({"success": True})
        except sqlite3.IntegrityError:
            return jsonify({"success": False, "msg": "该车牌已存在！"}), 400


# 新增：临停记录分页查询 API
@app.route('/api/records')
def get_records():
    if not session.get('is_admin'): return jsonify({"error": "未授权"}), 401
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()

    page = int(request.args.get('page', 1))
    search = request.args.get('search', '')
    limit = 8
    offset = (page - 1) * limit

    query = "SELECT id, plate_number, entry_time, exit_time, duration_hours, fee FROM parking_records"
    count_query = "SELECT COUNT(*) FROM parking_records"
    params = []

    if search:
        query += " WHERE plate_number LIKE ?"
        count_query += " WHERE plate_number LIKE ?"
        params = [f"%{search}%"]

    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    cursor.execute(query, params + [limit, offset])
    rows = cursor.fetchall()

    data = [{"id": r[0], "plate": r[1], "entry": r[2], "exit": r[3], "duration": r[4], "fee": r[5]} for r in rows]
    return jsonify({"data": data, "total": total})


# 新增：数据导出 CSV API
@app.route('/api/export')
def export_records():
    if not session.get('is_admin'): return "未授权", 401
    conn = sqlite3.connect('parking.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT id, plate_number, entry_time, exit_time, duration_hours, fee FROM parking_records")
    rows = cursor.fetchall()

    # 生成 CSV 文件
    si = StringIO()
    cw = csv.writer(si)
    # 写入表头 (utf-8-sig 是为了防止 Excel 打开乱码)
    cw.writerow(['记录ID', '车牌号', '入场时间', '出场时间', '停车时长(小时)', '收费(元)'])
    cw.writerows(rows)

    output = make_response(si.getvalue().encode('utf-8-sig'))
    output.headers["Content-Disposition"] = "attachment; filename=parking_records.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)