import sqlite3
import os

DB_FILE = 'parking.db'


def clear_all_data():
    if not os.path.exists(DB_FILE):
        print("数据库文件不存在，无需清理。")
        return

    try:
        # 连接数据库
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # 清空所有表的数据
        cursor.execute("DELETE FROM current_vehicles")
        cursor.execute("DELETE FROM parking_records")

        # 重置自增 ID（可选，保证流水号从1开始）
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='parking_records'")

        conn.commit()
        conn.close()
        print("✅ 成功！历史记录和在场车辆已全部清空。")
        print("⚠️ 注意：请务必刷新网页或重启 app.py 查看最新状态。")

    except Exception as e:
        print(f"❌ 清理失败：{e}")


if __name__ == '__main__':
    clear_all_data()