import sqlite3
from datetime import datetime, timedelta
from flask import Flask, request, render_template, redirect, session, jsonify
import hashlib
from functools import wraps

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
DATABASE = 'study_planner.db'

# ========== HELPER FUNCTIONS (DEFINE THESE FIRST) ==========

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

def init_db():
    conn = get_db()
    # Create tables if they don't exist
    with open('schema.sql', 'r') as f:
        conn.executescript(f.read())
    conn.close()

# ========== AI CLASS (DEFINE BEFORE USING) ==========

class SimpleStudyPlannerAI:
    """Simple AI for generating study schedules"""
    
    def estimate_topic_time(self, topic_difficulty, user_level):
        """
        Estimate time needed for a topic based on difficulty and user level
        Simple heuristic: Easy=1hr, Medium=2hrs, Hard=3hrs
        Adjust based on user level (1-10 scale)
        """
        base_times = {'easy': 1.0, 'medium': 2.0, 'hard': 3.0}
        base_time = base_times.get(topic_difficulty.lower(), 2.0)
        
        # Adjust based on user level (1-10, higher is better)
        # If user is good (8-10), reduce time by 30%
        # If user is weak (1-3), increase time by 50%
        if user_level >= 8:
            return base_time * 0.7
        elif user_level <= 3:
            return base_time * 1.5
        return base_time
    
    def calculate_priority_score(self, topic, subject_exam_date, days_until_exam):
        """
        Calculate priority score for a topic
        Higher score = higher priority
        """
        priority_weights = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        difficulty_weights = {
            'hard': 3,
            'medium': 2,
            'easy': 1
        }
        
        # Convert numeric priority to string if needed
        priority_level = topic.get('priority_level', 'medium')
        if isinstance(priority_level, int):
            if priority_level == 1:
                priority_level = 'high'
            elif priority_level == 2:
                priority_level = 'medium'
            else:
                priority_level = 'low'
        
        # Base priority from database
        priority_score = priority_weights.get(priority_level, 1)
        
        # Add difficulty weight
        difficulty = topic.get('difficulty', 'medium')
        priority_score += difficulty_weights.get(difficulty, 1)
        
        # Add urgency factor (closer to exam = higher priority)
        if days_until_exam > 0:
            urgency_factor = 30 / days_until_exam  # Inverse relationship
            priority_score += min(urgency_factor, 10)  # Cap at 10
        
        return priority_score
    
    def generate_schedule(self, user_id, start_date=None, days_ahead=30):
        """
        Main AI function to generate study schedule
        Simple algorithm: Sort topics by priority, allocate time slots
        """
        conn = get_db()
        
        # Get user preferences
        preferences = conn.execute(
            'SELECT * FROM user_schedule_preferences WHERE user_id = ?',
            (user_id,)
        ).fetchone()
        
        if not preferences:
            # Create default preferences
            conn.execute(
                'INSERT INTO user_schedule_preferences (user_id, daily_study_hours) VALUES (?, ?)',
                (user_id, 4)
            )
            conn.commit()
            daily_hours = 4
            preferred_slot = 'morning'
        else:
            daily_hours = preferences['daily_study_hours'] or 4
            preferred_slot = preferences['preferred_study_slot'] or 'morning'
        
        # Get all subjects for user
        subjects = conn.execute(
            'SELECT * FROM subjects WHERE user_id = ?',
            (user_id,)
        ).fetchall()
        
        if not subjects:
            conn.close()
            return {"error": "No subjects found. Please add subjects first."}
        
        # Get all topics from syllabus
        all_topics = []
        for subject in subjects:
            topics = conn.execute(
                '''SELECT st.*, s.exam_date, s.self_rating 
                   FROM syllabus_topics st 
                   JOIN subjects s ON st.subject_id = s.subject_id
                   WHERE st.subject_id = ? AND st.is_completed = 0
                   ORDER BY st.sequence_order''',
                (subject['subject_id'],)
            ).fetchall()
            
            for topic in topics:
                # Calculate days until exam
                days_until_exam = 30  # Default
                if topic['exam_date']:
                    try:
                        exam_date = datetime.strptime(topic['exam_date'], '%Y-%m-%d')
                        today = datetime.now()
                        days_until_exam = (exam_date - today).days
                        if days_until_exam < 0:
                            days_until_exam = 0
                    except:
                        days_until_exam = 30
                
                # Estimate time needed
                estimated_time = self.estimate_topic_time(
                    topic['difficulty'] or 'medium',
                    topic['self_rating'] or 5
                )
                
                # Calculate priority score
                priority_score = self.calculate_priority_score(
                    {
                        'priority_level': topic['priority_level'],
                        'difficulty': topic['difficulty']
                    },
                    topic['exam_date'],
                    days_until_exam
                )
                
                all_topics.append({
                    'topic_id': topic['topic_id'],
                    'subject_id': topic['subject_id'],
                    'topic_name': topic['topic_name'],
                    'estimated_time': estimated_time,
                    'priority_score': priority_score,
                    'difficulty': topic['difficulty'],
                    'exam_date': topic['exam_date']
                })
        
        # Sort topics by priority score (highest first)
        all_topics.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Generate schedule
        if not start_date:
            start_date = datetime.now().date()
        else:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        schedule = []
        current_date = start_date
        hours_allocated_today = 0
        days_scheduled = 0
        
        for topic in all_topics:
            topic_hours = topic['estimated_time']
            
            # If topic takes more than daily limit, split it
            while topic_hours > 0 and days_scheduled < days_ahead:
                # Check if we need to move to next day
                if hours_allocated_today + topic_hours > daily_hours:
                    # Allocate what we can today
                    allocate_today = daily_hours - hours_allocated_today
                    if allocate_today > 0:
                        # Create schedule entry for today
                        schedule.append({
                            'date': current_date.strftime('%Y-%m-%d'),
                            'topic_id': topic['topic_id'],
                            'topic_name': topic['topic_name'],
                            'subject_id': topic['subject_id'],
                            'hours': allocate_today,
                            'is_full_session': False
                        })
                        topic_hours -= allocate_today
                    
                    # Move to next day
                    current_date += timedelta(days=1)
                    hours_allocated_today = 0
                    days_scheduled += 1
                else:
                    # Allocate full topic today
                    schedule.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'topic_id': topic['topic_id'],
                        'topic_name': topic['topic_name'],
                        'subject_id': topic['subject_id'],
                        'hours': topic_hours,
                        'is_full_session': True
                    })
                    hours_allocated_today += topic_hours
                    topic_hours = 0
        
        conn.close()
        return schedule
    
    def adapt_schedule_based_on_mood(self, user_id, mood_score, energy_level):
        """
        Simple mood-based adaptation
        Low mood/energy → schedule easier topics or revision
        """
        conn = get_db()
        
        # Get today's schedule
        today = datetime.now().strftime('%Y-%m-%d')
        today_schedule = conn.execute(
            '''SELECT ss.*, st.difficulty 
               FROM study_schedules ss 
               JOIN syllabus_topics st ON ss.topic_id = st.topic_id
               WHERE ss.user_id = ? AND ss.scheduled_date = ? AND ss.is_completed = 0''',
            (user_id, today)
        ).fetchall()
        
        adaptations = []
        
        if mood_score <= 2 or energy_level <= 2:  # Low mood/energy
            # Suggest replacing hard topics with easy ones or revision
            for task in today_schedule:
                if task['difficulty'] == 'hard':
                    # Find an easier alternative topic
                    easy_topic = conn.execute(
                        '''SELECT * FROM syllabus_topics 
                           WHERE subject_id = ? AND difficulty = 'easy' AND is_completed = 0
                           LIMIT 1''',
                        (task['subject_id'],)
                    ).fetchone()
                    
                    if easy_topic:
                        adaptations.append({
                            'original_topic': task['topic_id'],
                            'suggested_topic': easy_topic['topic_id'],
                            'reason': 'Low energy day - suggesting easier topic',
                            'original_difficulty': 'hard',
                            'suggested_difficulty': 'easy'
                        })
        
        conn.close()
        return adaptations

# ========== FLASK ROUTES ==========

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect('/dashboard')
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']
        full_name = request.form['full_name']
        
        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (email, password_hash, username, full_name) VALUES (?, ?, ?, ?)',
                (email, hash_password(password), username, full_name)
            )
            conn.commit()
            
            # Get the user_id of the newly created user
            user = conn.execute('SELECT user_id FROM users WHERE email = ?', (email,)).fetchone()
            
            # Create default profile
            conn.execute(
                'INSERT INTO profiles (user_id) VALUES (?)',
                (user['user_id'],)
            )
            
            # Create default schedule preferences
            conn.execute(
                'INSERT INTO user_schedule_preferences (user_id) VALUES (?)',
                (user['user_id'],)
            )
            
            conn.commit()
            conn.close()
            return redirect('/login')
        except sqlite3.IntegrityError as e:
            conn.close()
            return f"Username or email already exists: {str(e)}"
        except Exception as e:
            conn.close()
            return f"Error: {str(e)}"
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE email = ? AND password_hash = ?',
            (email, hash_password(password))
        ).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            return redirect('/dashboard')
        else:
            return "Invalid credentials"
    
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get user info
    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    
    # Get subjects
    subjects = conn.execute(
        'SELECT * FROM subjects WHERE user_id = ? ORDER BY exam_date',
        (user_id,)
    ).fetchall()
    
    # Get today's schedule
    today = datetime.now().strftime('%Y-%m-%d')
    today_schedule = conn.execute(
        '''SELECT ss.*, s.subject_name, st.topic_name 
           FROM study_schedules ss
           LEFT JOIN subjects s ON ss.subject_id = s.subject_id
           LEFT JOIN syllabus_topics st ON ss.topic_id = st.topic_id
           WHERE ss.user_id = ? AND ss.scheduled_date = ?
           ORDER BY ss.start_time''',
        (user_id, today)
    ).fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                          user=user, 
                          subjects=subjects, 
                          today_schedule=today_schedule)

@app.route('/add_subject', methods=['POST'])
@login_required
def add_subject():
    user_id = session['user_id']
    subject_name = request.form['subject_name']
    exam_date = request.form.get('exam_date', None)
    self_rating = request.form.get('self_rating', 5)
    
    conn = get_db()
    try:
        conn.execute(
            '''INSERT INTO subjects (user_id, subject_name, exam_date, self_rating) 
               VALUES (?, ?, ?, ?)''',
            (user_id, subject_name, exam_date, self_rating)
        )
        conn.commit()
        conn.close()
        return redirect('/dashboard')
    except Exception as e:
        conn.close()
        return f"Error adding subject: {str(e)}"
@app.route('/delete_subject', methods=['POST'])
@login_required
def delete_subject():
    subject_id = request.form['subject_id']

    conn = get_db()
    try:
        conn.execute(
            'DELETE FROM subjects WHERE subject_id = ?',
            (subject_id,)
        )
        conn.commit()
        conn.close()
        return redirect('/dashboard')

    except Exception as e:
        conn.close()
        return f"Error deleting subject: {str(e)}"
@app.route('/edit_subject/<int:subject_id>', methods=['GET', 'POST'])
@login_required
def edit_subject(subject_id):
    conn = get_db()

    try:
        # UPDATE subject
        if request.method == 'POST':
            subject_name = request.form['subject_name']
            exam_date = request.form['exam_date']
            self_rating = request.form['self_rating']

            conn.execute(
                '''UPDATE subjects
                   SET subject_name = ?, exam_date = ?, self_rating = ?
                   WHERE subject_id = ?''',
                (subject_name, exam_date, self_rating, subject_id)
            )

            conn.commit()
            conn.close()
            return redirect('/dashboard')

        # GET subject for edit form
        subject = conn.execute(
            'SELECT * FROM subjects WHERE subject_id = ?',
            (subject_id,)
        ).fetchone()

        conn.close()

        if subject is None:
            return "Subject not found", 404

        return render_template('edit_subject.html', subject=subject)

    except Exception as e:
        conn.close()
        return f"Error editing subject: {str(e)}"

@app.route('/add_topic', methods=['POST'])
@login_required
def add_topic():
    subject_id = request.form['subject_id']
    topic_name = request.form['topic_name']
    difficulty = request.form['difficulty']
    priority = request.form['priority']
    
    conn = get_db()
    try:
        # Get the next sequence order
        max_order = conn.execute(
            'SELECT MAX(sequence_order) as max_order FROM syllabus_topics WHERE subject_id = ?',
            (subject_id,)
        ).fetchone()
        
        next_order = (max_order['max_order'] or 0) + 1
        
        conn.execute(
            '''INSERT INTO syllabus_topics (subject_id, topic_name, difficulty, priority_level, sequence_order) 
               VALUES (?, ?, ?, ?, ?)''',
            (subject_id, topic_name, difficulty, priority, next_order)
        )
        conn.commit()
        conn.close()
        return redirect('/dashboard')
    except Exception as e:
        conn.close()
        return f"Error adding topic: {str(e)}"

@app.route('/generate_schedule')
@login_required
def generate_schedule():
    user_id = session['user_id']
    
    # Use our AI planner
    planner = SimpleStudyPlannerAI()
    schedule = planner.generate_schedule(user_id)
    
    if 'error' in schedule:
        return schedule['error']
    
    # Save schedule to database
    conn = get_db()
    
    try:
        # Clear existing schedule for next 30 days
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=30)
        
        conn.execute(
            '''DELETE FROM study_schedules 
               WHERE user_id = ? AND scheduled_date BETWEEN ? AND ?''',
            (user_id, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        )
        
        # Add new schedule
        for item in schedule:
            # Convert hours to time slots (simplified)
            start_time = "09:00"  # Default morning slot
            end_hour = 9 + int(item['hours'])
            end_time = f"{end_hour:02d}:00"
            
            conn.execute(
                '''INSERT INTO study_schedules 
                   (user_id, subject_id, topic_id, scheduled_date, start_time, end_time) 
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (user_id, item['subject_id'], item['topic_id'], 
                 item['date'], start_time, end_time)
            )
        
        conn.commit()
        conn.close()
        return redirect('/dashboard')
    except Exception as e:
        conn.close()
        return f"Error generating schedule: {str(e)}"

@app.route('/update_progress', methods=['POST'])
@login_required
def update_progress():
    user_id = session['user_id']
    schedule_id = request.form['schedule_id']
    completion = request.form['completion']  # 'done', 'partial', 'skipped'
    
    conn = get_db()
    
    try:
        if completion == 'done':
            conn.execute(
                'UPDATE study_schedules SET is_completed = 1, completion_percentage = 100 WHERE schedule_id = ?',
                (schedule_id,)
            )
            
            # Mark topic as completed if all sessions for it are done
            schedule = conn.execute(
                'SELECT * FROM study_schedules WHERE schedule_id = ?',
                (schedule_id,)
            ).fetchone()
            
            if schedule and schedule['topic_id']:
                # Check if all sessions for this topic are completed
                remaining = conn.execute(
                    '''SELECT COUNT(*) as count FROM study_schedules 
                       WHERE topic_id = ? AND is_completed = 0 AND user_id = ?''',
                    (schedule['topic_id'], user_id)
                ).fetchone()
                
                if remaining['count'] == 0:
                    conn.execute(
                        'UPDATE syllabus_topics SET is_completed = 1, completed_date = DATE("now") WHERE topic_id = ?',
                        (schedule['topic_id'],)
                    )
        
        elif completion == 'partial':
            conn.execute(
                'UPDATE study_schedules SET completion_percentage = 50 WHERE schedule_id = ?',
                (schedule_id,)
            )
        else:  # skipped
            conn.execute(
                'UPDATE study_schedules SET is_completed = 0, completion_percentage = 0 WHERE schedule_id = ?',
                (schedule_id,)
            )
        
        conn.commit()
        conn.close()
        return redirect('/dashboard')
    except Exception as e:
        conn.close()
        return f"Error updating progress: {str(e)}"

@app.route('/log_mood', methods=['POST'])
@login_required
def log_mood():
    user_id = session['user_id']
    mood_score = request.form.get('mood_score', 3)
    energy_level = request.form.get('energy_level', 3)
    stress_level = request.form.get('stress_level', 3)
    
    conn = get_db()
    
    try:
        # Check if mood already logged today
        today = datetime.now().strftime('%Y-%m-%d')
        existing = conn.execute(
            'SELECT * FROM mood_tracking WHERE user_id = ? AND tracking_date = ?',
            (user_id, today)
        ).fetchone()
        
        if existing:
            conn.execute(
                '''UPDATE mood_tracking 
                   SET mood_score = ?, energy_level = ?, stress_level = ?
                   WHERE mood_id = ?''',
                (mood_score, energy_level, stress_level, existing['mood_id'])
            )
        else:
            conn.execute(
                '''INSERT INTO mood_tracking (user_id, mood_score, energy_level, stress_level) 
                   VALUES (?, ?, ?, ?)''',
                (user_id, mood_score, energy_level, stress_level)
            )
        
        # Get AI adaptation suggestions
        planner = SimpleStudyPlannerAI()
        adaptations = planner.adapt_schedule_based_on_mood(
            user_id, int(mood_score), int(energy_level)
        )
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Mood logged successfully',
            'adaptations': adaptations
        })
    except Exception as e:
        conn.close()
        return jsonify({
            'status': 'error',
            'message': f'Error logging mood: {str(e)}'
        })

@app.route('/get_ai_recommendations')
@login_required
def get_ai_recommendations():
    user_id = session['user_id']
    
    conn = get_db()
    
    try:
        # Simple AI recommendations based on progress
        # 1. Find weak areas
        weak_topics = conn.execute(
            '''SELECT st.*, s.subject_name 
               FROM syllabus_topics st
               JOIN subjects s ON st.subject_id = s.subject_id
               WHERE s.user_id = ? AND st.is_completed = 0 
               AND (st.difficulty = 'hard' OR st.priority_level = 1)
               ORDER BY s.exam_date
               LIMIT 3''',
            (user_id,)
        ).fetchall()
        
        # 2. Find topics due soon
        today = datetime.now().date()
        due_soon = conn.execute(
            '''SELECT st.*, s.subject_name, s.exam_date 
               FROM syllabus_topics st
               JOIN subjects s ON st.subject_id = s.subject_id
               WHERE s.user_id = ? AND st.is_completed = 0 
               AND s.exam_date IS NOT NULL
               AND DATE(s.exam_date) <= DATE(?, '+7 days')
               ORDER BY s.exam_date
               LIMIT 3''',
            (user_id, today.strftime('%Y-%m-%d'))
        ).fetchall()
        
        conn.close()
        
        recommendations = []
        
        # Add weak topics recommendations
        for topic in weak_topics:
            recommendations.append({
                'type': 'weak_area',
                'topic': topic['topic_name'],
                'subject': topic['subject_name'],
                'message': f"Focus on '{topic['topic_name']}' as it's a high priority topic in {topic['subject_name']}"
            })
        
        # Add due soon recommendations
        for topic in due_soon:
            if topic['exam_date']:
                try:
                    exam_date = datetime.strptime(topic['exam_date'], '%Y-%m-%d').date()
                    days_left = (exam_date - today).days
                    recommendations.append({
                        'type': 'due_soon',
                        'topic': topic['topic_name'],
                        'subject': topic['subject_name'],
                        'days_left': days_left,
                        'message': f"Revise '{topic['topic_name']}' for {topic['subject_name']} exam in {days_left} days"
                    })
                except:
                    continue
        
        return jsonify({'status': 'success', 'recommendations': recommendations})
    except Exception as e:
        conn.close()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ========== MAIN ==========

if __name__ == '__main__':
    # Create database tables if they don't exist
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Note: Database already exists or error: {e}")
    
    app.run(debug=True, port=5000)