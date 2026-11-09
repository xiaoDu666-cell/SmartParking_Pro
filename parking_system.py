import sqlite3
import time
from datetime import datetime

# 数据库文件名
DB_NAME = 'parking.db'


def init_db():
    """初始化数据库，建立两张表"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 1. 在场车辆表 (只记录当前还在停车场的车)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS current_cars (
            plate_number TEXT PRIMARY KEY,
            entry_time TEXT NOT NULL
        )
    ''')

    # 2. 历史流水表 (记录所有进出记录，用于后续大屏分析)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parking_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            duration_hours REAL NOT NULL,
            fee REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("数据库与数据表初始化完成！")


def calculate_fee(entry_time_str, exit_time_str):
    """
    计算停车费和时长
    规则：首小时5元，之后每小时3元，不足1小时按1小时算
    """
    fmt = '%Y-%m-%d %H:%M:%S'
    entry_time = datetime.strptime(entry_time_str, fmt)
    exit_time = datetime.strptime(exit_time_str, fmt)

    # 计算差值（秒）
    delta_seconds = (exit_time - entry_time).total_seconds()
    # 向上取整到小时
    hours = int(delta_seconds / 3600)
    if delta_seconds % 3600 > 0:
        hours += 1

    # 如果不到1小时，按1小时算
    if hours <= 1:
        fee = 5.0
    else:
        fee = 5.0 + (hours - 1) * 3.0

    return hours, fee


def enter_parking(plate_number):
    """车辆入场逻辑"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 检查是否已经在场
    cursor.execute("SELECT * FROM current_cars WHERE plate_number = ?", (plate_number,))
    if cursor.fetchone():
        print(f"[警告] 车辆 {plate_number} 已经在停车场内！")
        conn.close()
        return False

    # 记录入场时间
    entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("INSERT INTO current_cars (plate_number, entry_time) VALUES (?, ?)", (plate_number, entry_time))
    conn.commit()
    conn.close()
    print(f"[入场] 车辆 {plate_number} 入场成功！时间: {entry_time}")
    return True


def exit_parking(plate_number):
    """车辆出场逻辑"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 查找入场时间
    cursor.execute("SELECT entry_time FROM current_cars WHERE plate_number = ?", (plate_number,))
    result = cursor.fetchone()

    if not result:
        print(f"[错误] 未找到车辆 {plate_number} 的入场记录，可能未进场或已出场！")
        conn.close()
        return None

    entry_time = result[0]
    exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 计算费用
    hours, fee = calculate_fee(entry_time, exit_time)

    # 从在场表中删除
    cursor.execute("DELETE FROM current_cars WHERE plate_number = ?", (plate_number,))
    # 写入历史流水表
    cursor.execute('''
        INSERT INTO parking_records (plate_number, entry_time, exit_time, duration_hours, fee) 
        VALUES (?, ?, ?, ?, ?)
    ''', (plate_number, entry_time, exit_time, hours, fee))

    conn.commit()
    conn.close()

    print(f"[出场] 车辆 {plate_number} 出场！")
    print(f"       入场时间: {entry_time}")
    print(f"       出场时间: {exit_time}")
    print(f"       停车时长: {hours} 小时")
    print(f"       应收费用: {fee} 元")
    return fee


def show_current_cars():
    """显示在场车辆"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM current_cars")
    cars = cursor.fetchall()
    conn.close()

    print("\n--- 当前在场车辆 ---")
    if not cars:
        print("停车场为空")
    for car in cars:
        print(f"车牌: {car[0]}, 入场时间: {car[1]}")
    print("--------------------\n")

def check_in_parking(plate_number):
    """检查车辆是否在场（用于防止重复入场）"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM current_cars WHERE plate_number = ?", (plate_number,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# ================= 模拟测试 =================
if __name__ == '__main__':
    # 1. 初始化数据库
    init_db()

    # 2. 模拟测试（注意：为了测试计费，这里会伪造一条历史入场记录）
    print("\n--- 测试开始 ---")

    # 场景1：A车入场
    plate1 = "粤B·GK9689"
    enter_parking(plate1)

    # 模拟过了2.5小时（我们为了测试，强行修改数据库里的入场时间）
    print("\n(系统内部操作：强行将入场时间修改为2.5小时前，用于模拟)...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    fake_entry_time = datetime.fromtimestamp(time.time() - 2.5 * 3600).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("UPDATE current_cars SET entry_time = ? WHERE plate_number = ?", (fake_entry_time, plate1))
    conn.commit()
    conn.close()

    # 查看在场车辆
    show_current_cars()

    # 场景2：A车出场
    exit_parking(plate1)

    # 场景3：再次尝试让A车出场（测试异常情况）
    exit_parking(plate1)