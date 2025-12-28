import sqlite3

# Connect to SQLite database (it will be created if not exists)
conn = sqlite3.connect("study_planner.db")
cursor = conn.cursor()

# Enable foreign key constraints
cursor.execute("PRAGMA foreign_keys = ON;")

# -------------------- USERS TABLE --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    username TEXT UNIQUE,
    full_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active INTEGER DEFAULT 1
);
""")

# -------------------- PROFILES TABLE --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER UNIQUE NOT NULL,
    date_of_birth DATE,
    gender TEXT,
    phone_number TEXT,
    timezone TEXT DEFAULT 'UTC',
    preferred_language TEXT DEFAULT 'English',
    theme_preference TEXT DEFAULT 'light',
    font_size INTEGER DEFAULT 14,
    color_mode TEXT DEFAULT 'default',
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
""")

# -------------------- USER SCHEDULE PREFERENCES --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_schedule_preferences (
    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    daily_study_hours INTEGER DEFAULT 4,
    preferred_study_slot TEXT,
    break_length_minutes INTEGER DEFAULT 15,
    max_daily_load_hours INTEGER DEFAULT 8,
    sleep_time TIME,
    wake_up_time TIME,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
""")

# -------------------- USER GOALS --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_goals (
    goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    goal_type TEXT NOT NULL,
    description TEXT NOT NULL,
    target_date DATE,
    target_marks TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
""")

# -------------------- SUBJECTS --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subject_name TEXT NOT NULL,
    exam_date DATE,
    self_rating INTEGER,
    current_level TEXT,
    weak_areas TEXT,
    strong_areas TEXT,
    past_score REAL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
""")

# -------------------- SYLLABUS TOPICS --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS syllabus_topics (
    topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    topic_name TEXT NOT NULL,
    estimated_time_hours REAL,
    priority_level INTEGER DEFAULT 3,
    difficulty TEXT,
    sequence_order INTEGER,
    is_completed INTEGER DEFAULT 0,
    completed_date DATE,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
);
""")

# -------------------- STUDY SCHEDULES --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS study_schedules (
    schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subject_id INTEGER,
    topic_id INTEGER,
    scheduled_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL,
    study_method TEXT,
    is_completed INTEGER DEFAULT 0,
    completion_percentage INTEGER DEFAULT 0,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE SET NULL,
    FOREIGN KEY (topic_id) REFERENCES syllabus_topics(topic_id) ON DELETE SET NULL
);
""")

# -------------------- PROGRESS TRACKING --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS progress_tracking (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subject_id INTEGER,
    date DATE DEFAULT CURRENT_DATE,
    time_spent_minutes INTEGER,
    topics_covered INTEGER,
    self_assessment_score INTEGER,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE SET NULL
);
""")

# -------------------- MOOD TRACKING --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS mood_tracking (
    mood_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    tracking_date DATE DEFAULT CURRENT_DATE,
    mood_score INTEGER,
    energy_level INTEGER,
    stress_level INTEGER,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
""")

# -------------------- AI CONTENT --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS ai_content (
    content_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subject_id INTEGER,
    topic_id INTEGER,
    content_type TEXT NOT NULL,
    content_text TEXT NOT NULL,
    answer_text TEXT,
    difficulty TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE SET NULL,
    FOREIGN KEY (topic_id) REFERENCES syllabus_topics(topic_id) ON DELETE SET NULL
);
""")

# -------------------- NOTIFICATIONS --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS notifications (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    notification_type TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    scheduled_time TIMESTAMP,
    is_sent INTEGER DEFAULT 0,
    sent_time TIMESTAMP,
    channel TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
""")

# -------------------- USER CONSTRAINTS --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_constraints (
    constraint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    constraint_date DATE NOT NULL,
    constraint_type TEXT NOT NULL,
    description TEXT,
    start_time TIME,
    end_time TIME,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);
""")

# -------------------- STUDY METHOD PREFERENCES --------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS study_methods_preferences (
    method_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    subject_id INTEGER,
    preferred_method TEXT NOT NULL,
    effectiveness_rating INTEGER,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE SET NULL
);
""")

# -------------------- INDEXES --------------------
cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_profiles_user_id ON profiles(user_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_schedules_user_date ON study_schedules(user_id, scheduled_date);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_subject ON syllabus_topics(subject_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_subjects_user ON subjects(user_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_user_date ON mood_tracking(user_id, tracking_date);")

# Commit and close
conn.commit()
conn.close()

print("✅ Database and tables created successfully!")
