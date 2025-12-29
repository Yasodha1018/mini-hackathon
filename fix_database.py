import sqlite3

DATABASE = 'study_planner.db'

def fix_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Add missing columns
    try:
        # Add study_method column if not exists
        cursor.execute("PRAGMA table_info(user_schedule_preferences)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'study_method' not in columns:
            cursor.execute('ALTER TABLE user_schedule_preferences ADD COLUMN study_method TEXT')
            print("Added study_method column")
        
        # Add missing columns to other tables
        cursor.execute("PRAGMA table_info(study_schedules)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'study_method' not in columns:
            cursor.execute('ALTER TABLE study_schedules ADD COLUMN study_method TEXT')
            print("Added study_method to study_schedules")
            
    except Exception as e:
        print(f"Error fixing database: {e}")
    
    conn.commit()
    conn.close()
    print("✅ Database fixed successfully!")

if __name__ == "__main__":
    fix_database()