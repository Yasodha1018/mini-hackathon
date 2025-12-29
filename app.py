
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta, date
from flask import Flask, request, render_template, redirect, session, jsonify, send_file
from functools import wraps
import os
from werkzeug.utils import secure_filename
import pytz
from dateutil import parser
import random
import uuid
import json
import time
from typing import List, Dict, Any
import requests
app = Flask(__name__)
app.secret_key = 'ai_study_planner_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

DATABASE = 'study_planner.db'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf', 'txt', 'doc', 'docx']

# ========== DATABASE INITIALIZATION ==========

def init_db():
    conn = get_db()
    # Your existing SQL schema will be used
    conn.close()
    print("Database initialized")



# ========== AI API CONFIGURATION ==========

class AIAPIConfig:
    """Configuration for AI APIs"""
    
    # Google Gemini API (Free - up to 60 requests per minute)
    GEMINI_API_KEY = "AIzaSyBI1cCEO5bvlaW5e86z0mEVTdNImCaH_1U"  # Replace with your key
    GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    # OpenRouter (Access to multiple models including Claude, GPT-4, etc.)
    OPENROUTER_API_KEY = "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"  # Get from https://openrouter.ai
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Hugging Face Inference API (Free)
    HUGGINGFACE_API_KEY = "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"  # Get from https://huggingface.co
    HUGGINGFACE_URL = "https://api-inference.huggingface.co/models"
    
    # Local Ollama (If installed locally)
    OLLAMA_URL = "http://localhost:11434/api/generate"

# ========== AI TUTOR CLASS ==========

class AITutor:
    """Intelligent AI Tutor using multiple AI APIs"""
    
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.config = AIAPIConfig()
        
    def get_user_context(self, topic_id=None):
        """Get user's learning context from database"""
        conn = get_db()
        
        context = {
            'subjects': [],
            'topics': [],
            'progress': {},
            'preferences': {}
        }
        
        if self.user_id:
            # Get user's subjects and topics
            subjects = conn.execute('''
                SELECT s.subject_name, s.self_rating, s.weak_areas, s.strong_areas,
                       st.topic_name, st.difficulty, st.is_completed
                FROM subjects s
                LEFT JOIN syllabus_topics st ON s.subject_id = st.subject_id
                WHERE s.user_id = ?
            ''', (self.user_id,)).fetchall()
            
            for subject in subjects:
                context['subjects'].append({
                    'name': subject['subject_name'],
                    'self_rating': subject['self_rating'],
                    'weak_areas': subject['weak_areas'],
                    'strong_areas': subject['strong_areas']
                })
                
                if subject['topic_name']:
                    context['topics'].append({
                        'name': subject['topic_name'],
                        'difficulty': subject['difficulty'],
                        'completed': subject['is_completed']
                    })
            
            # Get topic-specific context if provided
            if topic_id:
                topic = conn.execute('''
                    SELECT st.*, s.subject_name 
                    FROM syllabus_topics st
                    JOIN subjects s ON st.subject_id = s.subject_id
                    WHERE st.topic_id = ?
                ''', (topic_id,)).fetchone()
                
                if topic:
                    context['current_topic'] = {
                        'name': topic['topic_name'],
                        'subject': topic['subject_name'],
                        'difficulty': topic['difficulty'],
                        'description': topic.get('description', '')
                    }
            
            # Get study preferences
            preferences = conn.execute('''
                SELECT preferred_study_slot, study_method 
                FROM user_schedule_preferences up
                LEFT JOIN study_methods_preferences sm ON up.user_id = sm.user_id
                WHERE up.user_id = ?
            ''', (self.user_id,)).fetchone()
            
            if preferences:
                context['preferences'] = {
                    'preferred_time': preferences['preferred_study_slot'],
                    'study_method': preferences['study_method']
                }
        
        conn.close()
        return context
    
    def ask_ai_tutor(self, question: str, topic_id: str = None) -> Dict:
        """Ask AI tutor any question related to studies"""
        
        # Get user context for personalized response
        context = self.get_user_context(topic_id)
        
        # Construct prompt with context
        prompt = f"""You are an AI Study Tutor helping a student. 
        
        Student Context:
        - Subjects: {[s['name'] for s in context['subjects']]}
        - Learning Level: {context['subjects'][0]['self_rating']/10 if context['subjects'] else 'Unknown'}
        - Current Topics: {[t['name'] for t in context['topics'][:5]]}
        
        Question: {question}
        
        Please provide:
        1. Clear, step-by-step explanation
        2. Examples if relevant
        3. Common mistakes to avoid
        4. Practice suggestions
        5. Related topics to study
        
        Keep the response educational, friendly, and at an appropriate level for the student."""
        
        try:
            # Try Gemini API first
            response = self._call_gemini_api(prompt)
        except Exception as e:
            print(f"Gemini API failed: {e}")
            # Fallback to local logic
            response = self._generate_fallback_response(question, context)
        
        return response
    
    def generate_smart_revision_strategy(self, topic_id: str) -> Dict:
        """Generate personalized revision strategy using AI"""
        
        conn = get_db()
        topic = conn.execute('''
            SELECT st.*, s.subject_name, s.exam_date
            FROM syllabus_topics st
            JOIN subjects s ON st.subject_id = s.subject_id
            WHERE st.topic_id = ?
        ''', (topic_id,)).fetchone()
        
        if not topic:
            conn.close()
            return {"error": "Topic not found"}
        
        # Get user's study history
        progress = conn.execute('''
            SELECT * FROM progress_tracking 
            WHERE user_id = ? 
            ORDER BY date DESC 
            LIMIT 10
        ''', (self.user_id,)).fetchall()
        
        conn.close()
        
        # Construct AI prompt
        prompt = f"""Generate a personalized revision strategy for a student.
        
        Topic: {topic['topic_name']}
        Subject: {topic['subject_name']}
        Difficulty: {topic['difficulty']}
        Exam Date: {topic['exam_date'] or 'Not specified'}
        
        Student's Study History: {len(progress)} recent sessions
        
        Create a 7-day revision plan including:
        1. Day-by-day activities
        2. Recommended study methods (active recall, spaced repetition, etc.)
        3. Time allocation per session
        4. Practice exercises focus areas
        5. Self-assessment checkpoints
        6. Tips for retention
        
        Make it practical and tailored to the topic difficulty."""
        
        try:
            # Call AI API
            strategy = self._call_openrouter_api(prompt, model="meta-llama/llama-3.1-8b-instruct")
        except:
            # Generate basic strategy
            strategy = self._generate_basic_revision_strategy(topic)
        
        return strategy
    
    def generate_ai_flashcards(self, topic_id: str, count: int = 10) -> List[Dict]:
        """Generate AI-powered flashcards for a topic"""
        
        conn = get_db()
        topic = conn.execute('''
            SELECT st.*, s.subject_name
            FROM syllabus_topics st
            JOIN subjects s ON st.subject_id = s.subject_id
            WHERE st.topic_id = ?
        ''', (topic_id,)).fetchone()
        
        if not topic:
            conn.close()
            return []
        
        # Get related topics for context
        related_topics = conn.execute('''
            SELECT topic_name FROM syllabus_topics 
            WHERE subject_id = ? AND topic_id != ?
            LIMIT 5
        ''', (topic['subject_id'], topic_id)).fetchall()
        
        conn.close()
        
        prompt = f"""Generate {count} educational flashcards for the topic: {topic['topic_name']}
        
        Subject: {topic['subject_name']}
        Difficulty: {topic['difficulty']}
        Related Topics: {[t['topic_name'] for t in related_topics]}
        
        Format each flashcard as:
        Front: Question or term
        Back: Detailed explanation, definition, or answer
        
        Include:
        - Key concepts and definitions
        - Important formulas/theorems (if applicable)
        - Application examples
        - Common misconceptions
        
        Make the cards suitable for spaced repetition learning."""
        
        try:
            flashcards = self._call_gemini_api(prompt, is_flashcards=True)
        except:
            flashcards = self._generate_basic_flashcards(topic, count)
        
        # Save to database
        self._save_flashcards_to_db(topic_id, flashcards)
        
        return flashcards
    
    def generate_practice_questions(self, topic_id: str, count: int = 5, difficulty: str = "mixed") -> List[Dict]:
        """Generate AI practice questions with solutions"""
        
        conn = get_db()
        topic = conn.execute('''
            SELECT st.*, s.subject_name
            FROM syllabus_topics st
            JOIN subjects s ON st.subject_id = s.subject_id
            WHERE st.topic_id = ?
        ''', (topic_id,)).fetchone()
        
        if not topic:
            conn.close()
            return []
        
        # Get past performance data
        past_questions = conn.execute('''
            SELECT * FROM ai_content 
            WHERE user_id = ? AND topic_id = ? AND content_type = 'question'
            ORDER BY created_at DESC 
            LIMIT 5
        ''', (self.user_id, topic_id)).fetchall()
        
        conn.close()
        
        prompt = f"""Generate {count} practice questions for: {topic['topic_name']}
        
        Subject: {topic['subject_name']}
        Topic Difficulty: {topic['difficulty']}
        Desired Question Difficulty: {difficulty}
        
        Format each question as:
        1. Question text (clear and specific)
        2. Multiple choice options (A, B, C, D) if applicable
        3. Correct answer
        4. Step-by-step solution
        5. Explanation of concepts tested
        6. Difficulty rating (Easy/Medium/Hard)
        7. Time estimate for solving
        
        Include a mix of:
        - Conceptual understanding questions
        - Problem-solving questions
        - Application-based questions
        - Critical thinking questions
        
        Make questions progressively challenging."""
        
        try:
            questions = self._call_openrouter_api(prompt, model="openai/gpt-3.5-turbo", is_questions=True)
        except:
            questions = self._generate_basic_practice_questions(topic, count, difficulty)
        
        # Save to database
        self._save_questions_to_db(topic_id, questions)
        
        return questions
    
    def get_next_topic_recommendation(self) -> Dict:
        """AI recommendation for next topic to study"""
        
        context = self.get_user_context()
        
        prompt = f"""Based on the student's learning profile, recommend the next topic to study.
        
        Student Profile:
        - Subjects: {[s['name'] for s in context['subjects']]}
        - Completed Topics: {[t['name'] for t in context['topics'] if t['completed']]}
        - Pending Topics: {[t['name'] for t in context['topics'] if not t['completed']]}
        - Self Ratings: {[f"{s['name']}: {s['self_rating']}/10" for s in context['subjects']]}
        - Weak Areas: {context['subjects'][0]['weak_areas'] if context['subjects'] else 'None specified'}
        
        Recommend ONE topic to study next with:
        1. Why this topic is important now
        2. How it connects to already learned topics
        3. Estimated time to master
        4. Best study methods for this topic
        5. Expected learning outcomes
        
        Consider:
        - Prerequisite knowledge
        - Upcoming exams/deadlines
        - Learning progression
        - Student's strengths/weaknesses"""
        
        try:
            recommendation = self._call_gemini_api(prompt)
        except:
            recommendation = self._get_basic_next_topic(context)
        
        return recommendation
    
    def _call_gemini_api(self, prompt: str, is_flashcards: bool = False) -> Any:
        """Call Google Gemini API"""
        
        headers = {
            "Content-Type": "application/json",
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }
        
        # Add API key to URL
        url = f"{self.config.GEMINI_URL}?key={self.config.GEMINI_API_KEY}"
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if is_flashcards:
                return self._parse_flashcards_from_response(result)
            return self._parse_response(result)
        else:
            raise Exception(f"Gemini API error: {response.status_code}")
    
    def _call_openrouter_api(self, prompt: str, model: str = "meta-llama/llama-3.1-8b-instruct", is_questions: bool = False) -> Any:
        """Call OpenRouter API (multiple models)"""
        
        headers = {
            "Authorization": f"Bearer {self.config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",  # Your site URL
            "X-Title": "AI Study Planner"
        }
        
        data = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(self.config.OPENROUTER_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if is_questions:
                return self._parse_questions_from_response(result)
            return self._parse_response(result)
        else:
            raise Exception(f"OpenRouter API error: {response.status_code}")
    
    def _call_huggingface_api(self, model: str, prompt: str) -> Any:
        """Call Hugging Face Inference API"""
        
        url = f"{self.config.HUGGINGFACE_URL}/{model}"
        headers = {"Authorization": f"Bearer {self.config.HUGGINGFACE_API_KEY}"}
        
        response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Hugging Face API error: {response.status_code}")
    
    def _call_ollama_api(self, prompt: str, model: str = "llama2") -> Any:
        """Call local Ollama API (if installed)"""
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(self.config.OLLAMA_URL, json=data, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def _parse_response(self, api_response: Dict) -> Dict:
        """Parse AI API response"""
        try:
            # Gemini format
            if 'candidates' in api_response:
                text = api_response['candidates'][0]['content']['parts'][0]['text']
            # OpenRouter format
            elif 'choices' in api_response:
                text = api_response['choices'][0]['message']['content']
            # Ollama format
            elif 'response' in api_response:
                text = api_response['response']
            else:
                text = str(api_response)
            
            return {
                'response': text,
                'source': 'ai_api',
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {
                'response': str(api_response),
                'source': 'ai_api_raw',
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_flashcards_from_response(self, response: Dict) -> List[Dict]:
        """Parse flashcards from AI response"""
        # This would parse the structured response
        # For now, return basic format
        return [
            {
                'id': str(uuid.uuid4())[:8],
                'front': f"Flashcard {i+1} front",
                'back': f"Flashcard {i+1} back - Detailed explanation",
                'difficulty': 'medium',
                'topic': 'General'
            }
            for i in range(5)
        ]
    
    def _parse_questions_from_response(self, response: Dict) -> List[Dict]:
        """Parse practice questions from AI response"""
        return [
            {
                'id': str(uuid.uuid4())[:8],
                'question': f"Sample question {i+1}?",
                'options': ['A', 'B', 'C', 'D'],
                'correct_answer': 'A',
                'explanation': 'Detailed explanation',
                'difficulty': 'medium',
                'time_estimate': '5 minutes'
            }
            for i in range(5)
        ]
    
    def _generate_fallback_response(self, question: str, context: Dict) -> Dict:
        """Generate fallback response when API fails"""
        return {
            'response': f"I understand you're asking about: {question}\n\nBased on your learning context, here's what I suggest:\n\n1. Review the key concepts first\n2. Try solving related practice problems\n3. Use active recall techniques\n4. Seek additional resources if needed\n\nRemember to take regular breaks while studying!",
            'source': 'fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_basic_revision_strategy(self, topic: Dict) -> Dict:
        """Generate basic revision strategy"""
        return {
            'strategy': f"""
            Revision Strategy for: {topic['topic_name']}
            
            Day 1-2: Foundation Review
            - Review key concepts and definitions
            - Create summary notes
            - Watch explanatory videos
            
            Day 3-4: Practice Phase
            - Solve basic problems
            - Work through examples
            - Identify weak areas
            
            Day 5-6: Advanced Application
            - Solve complex problems
            - Apply concepts to new scenarios
            - Timed practice sessions
            
            Day 7: Final Review
            - Quick revision of all concepts
            - Mock test
            - Review mistakes
            
            Tips:
            - Use spaced repetition
            - Teach concepts to someone else
            - Create mind maps
            """,
            'estimated_time': '7 days',
            'difficulty_level': topic['difficulty']
        }
    
    def _generate_basic_flashcards(self, topic: Dict, count: int) -> List[Dict]:
        """Generate basic flashcards"""
        flashcards = []
        for i in range(min(count, 10)):
            flashcards.append({
                'id': str(uuid.uuid4())[:8],
                'front': f"Key concept {i+1} from {topic['topic_name']}",
                'back': f"Detailed explanation of concept {i+1} with examples",
                'difficulty': topic['difficulty'],
                'topic': topic['topic_name'],
                'subject': topic['subject_name']
            })
        return flashcards
    
    def _generate_basic_practice_questions(self, topic: Dict, count: int, difficulty: str) -> List[Dict]:
        """Generate basic practice questions"""
        questions = []
        for i in range(min(count, 10)):
            questions.append({
                'id': str(uuid.uuid4())[:8],
                'question': f"Sample question about {topic['topic_name']}?",
                'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                'correct_answer': 'Option A',
                'explanation': 'This is the correct answer because...',
                'difficulty': difficulty,
                'time_estimate': '5-10 minutes',
                'topic': topic['topic_name']
            })
        return questions
    
    def _get_basic_next_topic(self, context: Dict) -> Dict:
        """Get basic next topic recommendation"""
        pending_topics = [t for t in context['topics'] if not t['completed']]
        if pending_topics:
            topic = pending_topics[0]
            return {
                'recommended_topic': topic['name'],
                'reason': 'Next in sequence',
                'estimated_time': '2-3 hours',
                'study_methods': ['reading', 'practice problems'],
                'prerequisites': 'Basic understanding of subject'
            }
        return {'recommended_topic': 'No pending topics', 'reason': 'All topics completed'}
    
    def _save_flashcards_to_db(self, topic_id: str, flashcards: List[Dict]):
        """Save generated flashcards to database"""
        conn = get_db()
        for card in flashcards:
            conn.execute('''
                INSERT INTO ai_content 
                (user_id, topic_id, content_type, content_text, answer_text, difficulty)
                VALUES (?, ?, 'flashcard', ?, ?, ?)
            ''', (self.user_id, topic_id, card['front'], card['back'], card.get('difficulty', 'medium')))
        conn.commit()
        conn.close()
    
    def _save_questions_to_db(self, topic_id: str, questions: List[Dict]):
        """Save generated questions to database"""
        conn = get_db()
        for q in questions:
            conn.execute('''
                INSERT INTO ai_content 
                (user_id, topic_id, content_type, content_text, answer_text, difficulty)
                VALUES (?, ?, 'question', ?, ?, ?)
            ''', (self.user_id, topic_id, q['question'], q['explanation'], q.get('difficulty', 'medium')))
        conn.commit()
        conn.close()


# ========== AI STUDY PLANNER CLASS ==========

class AdvancedStudyPlannerAI:
    def __init__(self):
        self.study_methods = {
            'reading': {'time_factor': 1.0, 'efficiency': 0.7},
            'videos': {'time_factor': 0.8, 'efficiency': 0.8},
            'problems': {'time_factor': 1.2, 'efficiency': 0.9},
            'learning': {'time_factor': 1.0, 'efficiency': 0.85}
        }
    
    def calculate_topic_complexity(self, topic_name, subject_area):
        """Simple AI to estimate topic complexity"""
        complexity_keywords = {
            'hard': ['calculus', 'quantum', 'thermodynamics', 'organic', 'algorithm'],
            'medium': ['functions', 'waves', 'electro', 'inorganic', 'data structures'],
            'easy': ['basics', 'introduction', 'simple', 'fundamental', 'overview']
        }
        
        topic_lower = topic_name.lower()
        for difficulty, keywords in complexity_keywords.items():
            if any(keyword in topic_lower for keyword in keywords):
                return difficulty
        return 'medium'
    
    def estimate_study_time(self, topic_name, difficulty, user_level, study_method='reading'):
        """AI time estimation based on multiple factors"""
        base_times = {'easy': 1.5, 'medium': 3.0, 'hard': 5.0}
        
        # Ensure valid inputs
        difficulty = difficulty or 'medium'
        user_level = user_level or 5
        study_method = study_method or 'reading'
        
        # Adjust based on user level (1-10, higher is better)
        level_factor = 1.2 - (user_level * 0.1)  # 1.2 for level 1, 0.2 for level 10
        
        # Adjust based on study method
        method_info = self.study_methods.get(study_method, self.study_methods['reading'])
        method_factor = method_info['time_factor']
        
        # Calculate estimated time
        estimated_hours = base_times.get(difficulty, 3.0) * level_factor * method_factor
        
        return round(max(1.0, estimated_hours), 1)  # Minimum 1 hour
    
    def generate_priority_score(self, topic, exam_date, current_date, user_rating):
        """Calculate priority score for intelligent scheduling"""
        # Ensure valid inputs
        user_rating = user_rating or 5
        
        # Base score from user rating (1-10 scale)
        base_score = user_rating
        
        # Urgency factor (days until exam)
        if exam_date:
            try:
                days_until_exam = (exam_date - current_date).days
                if days_until_exam > 0:
                    urgency_factor = 100 / days_until_exam
                else:
                    urgency_factor = 100  # Exam is today or passed
            except:
                urgency_factor = 1
        else:
            urgency_factor = 1
        
        # Difficulty factor
        difficulty_factors = {'easy': 1, 'medium': 2, 'hard': 3}
        difficulty = topic.get('difficulty', 'medium')
        difficulty_factor = difficulty_factors.get(difficulty, 1)
        
        # Calculate final priority score
        priority_score = (base_score * 0.3) + (urgency_factor * 0.5) + (difficulty_factor * 0.2)
        
        return priority_score
    
    def create_smart_schedule(self, user_id, start_date=None, days_ahead=30):
        """Main AI scheduling algorithm"""
        conn = get_db()
        
        # Get user preferences and constraints
        user_prefs = conn.execute(
            'SELECT * FROM user_schedule_preferences WHERE user_id = ?', 
            (user_id,)
        ).fetchone()
        
        # Get user's preferred study slots with defaults
        daily_study_hours = user_prefs['daily_study_hours'] if user_prefs and user_prefs['daily_study_hours'] else 4
        preferred_slot = user_prefs['preferred_study_slot'] if user_prefs and user_prefs['preferred_study_slot'] else 'morning'
        
        # Get all subjects with topics
        subjects = conn.execute('''
            SELECT s.*, 
                   GROUP_CONCAT(st.topic_id) as topic_ids,
                   GROUP_CONCAT(st.topic_name) as topic_names,
                   GROUP_CONCAT(st.difficulty) as difficulties,
                   GROUP_CONCAT(st.priority_level) as priorities
            FROM subjects s
            LEFT JOIN syllabus_topics st ON s.subject_id = st.subject_id AND st.is_completed = 0
            WHERE s.user_id = ?
            GROUP BY s.subject_id
        ''', (user_id,)).fetchall()
        
        if not subjects:
            conn.close()
            return {"error": "No subjects found. Please add subjects and topics first."}
        
        # Get user constraints
        constraints = conn.execute(
            'SELECT * FROM user_constraints WHERE user_id = ? AND constraint_date >= ?',
            (user_id, datetime.now().strftime('%Y-%m-%d'))
        ).fetchall()
        
        # Parse topics and calculate priorities
        all_topics = []
        current_date = datetime.now().date()
        
        for subject in subjects:
            if subject['topic_ids']:
                topic_ids = subject['topic_ids'].split(',') if subject['topic_ids'] else []
                topic_names = subject['topic_names'].split(',') if subject['topic_names'] else []
                difficulties = subject['difficulties'].split(',') if subject['difficulties'] else []
                priorities = subject['priorities'].split(',') if subject['priorities'] else []
                
                exam_date = None
                if subject['exam_date']:
                    try:
                        exam_date = datetime.strptime(subject['exam_date'], '%Y-%m-%d').date()
                    except:
                        pass
                
                for i in range(len(topic_ids)):
                    # Ensure we have valid values
                    topic_name = topic_names[i] if i < len(topic_names) else f"Topic {i+1}"
                    difficulty = difficulties[i] if i < len(difficulties) else 'medium'
                    priority = int(priorities[i]) if i < len(priorities) and priorities[i].isdigit() else 2
                    
                    # Calculate time needed
                    time_needed = self.estimate_study_time(
                        topic_name,
                        difficulty,
                        subject['self_rating'] or 5,
                        'reading'  # Default method
                    )
                    
                    # Calculate priority score
                    priority_score = self.generate_priority_score(
                        {
                            'difficulty': difficulty,
                            'priority': priority
                        },
                        exam_date,
                        current_date,
                        subject['self_rating'] or 5
                    )
                    
                    all_topics.append({
                        'topic_id': topic_ids[i],
                        'topic_name': topic_name,
                        'subject_id': subject['subject_id'],
                        'subject_name': subject['subject_name'],
                        'time_needed': time_needed,
                        'priority_score': priority_score,
                        'difficulty': difficulty,
                        'exam_date': subject['exam_date']
                    })
        
        # Sort topics by priority (highest first)
        all_topics.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Define time slots based on preference
        time_slots = self.get_time_slots(preferred_slot, daily_study_hours)
        
        # Generate schedule
        schedule = []
        current_schedule_date = current_date if not start_date else datetime.strptime(start_date, '%Y-%m-%d').date()
        days_scheduled = 0
        
        # Apply constraints
        constraint_dates = {c['constraint_date'] for c in constraints}
        
        topic_index = 0
        while days_scheduled < days_ahead and topic_index < len(all_topics):
            # Skip constrained dates
            if current_schedule_date.strftime('%Y-%m-%d') in constraint_dates:
                current_schedule_date += timedelta(days=1)
                continue
            
            day_schedule = []
            total_hours = 0
            
            for slot_name, (start_time, end_time) in time_slots.items():
                if topic_index >= len(all_topics):
                    break
                
                topic = all_topics[topic_index]
                slot_duration = 2.0  # Default 2-hour slots
                
                # Adjust slot duration based on topic difficulty
                if topic['difficulty'] == 'hard':
                    slot_duration = 1.5  # Shorter slots for hard topics
                elif topic['difficulty'] == 'easy':
                    slot_duration = 2.5  # Longer slots for easy topics
                
                # Check if we can fit this topic
                if total_hours + slot_duration <= daily_study_hours:
                    day_schedule.append({
                        'slot_name': slot_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'topic_id': topic['topic_id'],
                        'topic_name': topic['topic_name'],
                        'subject_name': topic['subject_name'],
                        'subject_id': topic['subject_id'],
                        'duration': slot_duration,
                        'difficulty': topic['difficulty'],
                        'study_method': self.recommend_study_method(topic['difficulty'])
                    })
                    
                    total_hours += slot_duration
                    topic['time_needed'] -= slot_duration
                    
                    if topic['time_needed'] <= 0:
                        topic_index += 1
                else:
                    break
            
            if day_schedule:
                schedule.append({
                    'date': current_schedule_date.strftime('%Y-%m-%d'),
                    'day_name': current_schedule_date.strftime('%A'),
                    'total_hours': total_hours,
                    'slots': day_schedule,
                    'is_constrained': False
                })
            
            current_schedule_date += timedelta(days=1)
            days_scheduled += 1
        
        conn.close()
        return schedule
    
    def get_time_slots(self, preferred_slot, daily_hours):
        """Generate time slots based on user preference"""
        # Ensure preferred_slot is not None
        if not preferred_slot:
            preferred_slot = 'morning'
        
        slots = {
            'morning': [('08:00', '10:00'), ('10:15', '12:15')],
            'afternoon': [('13:00', '15:00'), ('15:15', '17:15')],
            'evening': [('18:00', '20:00'), ('20:15', '22:15')],
            'night': [('20:00', '22:00'), ('22:15', '00:15')]
        }
        
        # Get slots based on preference, default to morning
        if preferred_slot.lower() in slots:
            selected_slots = slots[preferred_slot.lower()]
        else:
            selected_slots = slots['morning']
        
        # Convert to dictionary format
        time_slots = {}
        slot_count = min(3, int(daily_hours/2))
        
        for i, (start, end) in enumerate(selected_slots[:slot_count]):
            time_slots[f'{preferred_slot.capitalize()} Slot {i+1}'] = (start, end)
        
        return time_slots
    
    def recommend_study_method(self, difficulty):
        """Recommend study method based on topic difficulty"""
        recommendations = {
            'easy': ['reading', 'videos'],
            'medium': ['videos', 'problems', 'learning'],
            'hard': ['problems', 'learning', 'videos']
        }
        
        difficulty = difficulty or 'medium'
        methods = recommendations.get(difficulty, ['learning'])
        
        import random
        return random.choice(methods)
    
    def generate_practice_questions(self, topic_name, difficulty, count=5):
        """Generate practice questions for a topic"""
        import random
        import uuid
        
        questions = []
        question_types = {
            'easy': ['multiple choice', 'true/false', 'fill in the blanks'],
            'medium': ['short answer', 'matching', 'diagram labeling'],
            'hard': ['essay', 'problem solving', 'case study']
        }
        
        difficulty = difficulty or 'medium'
        q_types = question_types.get(difficulty, ['short answer'])
        
        for i in range(count):
            q_type = random.choice(q_types)
            questions.append({
                'id': str(uuid.uuid4())[:8],
                'question': f"{topic_name} - {q_type.capitalize()} Question {i+1}",
                'type': q_type,
                'difficulty': difficulty,
                'hint': f"Review the key concepts of {topic_name}",
                'explanation': f"This question tests your understanding of {topic_name}"
            })
        
        return questions
    
    def adapt_schedule_for_mood(self, user_id, mood_data):
        """Adapt schedule based on user mood"""
        conn = get_db()
        
        mood_score = mood_data.get('mood_score', 3)
        energy_level = mood_data.get('energy_level', 3)
        stress_level = mood_data.get('stress_level', 3)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get today's schedule
        today_schedule = conn.execute('''
            SELECT ss.*, st.difficulty, st.topic_name, s.subject_name
            FROM study_schedules ss
            JOIN syllabus_topics st ON ss.topic_id = st.topic_id
            JOIN subjects s ON ss.subject_id = s.subject_id
            WHERE ss.user_id = ? AND ss.scheduled_date = ? AND ss.is_completed = 0
        ''', (user_id, today)).fetchall()
        
        adaptations = []
        
        # Low energy or high stress - suggest easier topics
        if energy_level <= 2 or stress_level >= 4:
            for task in today_schedule:
                if task['difficulty'] == 'hard':
                    # Find easier alternative
                    easy_topic = conn.execute('''
                        SELECT st.* FROM syllabus_topics st
                        JOIN subjects s ON st.subject_id = s.subject_id
                        WHERE s.user_id = ? AND st.difficulty = 'easy' 
                        AND st.is_completed = 0 AND st.subject_id = ?
                        LIMIT 1
                    ''', (user_id, task['subject_id'])).fetchone()
                    
                    if easy_topic:
                        adaptations.append({
                            'action': 'replace',
                            'original': {
                                'topic_id': task['topic_id'],
                                'topic_name': task['topic_name'],
                                'difficulty': 'hard'
                            },
                            'suggestion': {
                                'topic_id': easy_topic['topic_id'],
                                'topic_name': easy_topic['topic_name'],
                                'difficulty': 'easy'
                            },
                            'reason': 'Low energy/high stress day - suggesting easier topic'
                        })
        
        # High energy - can handle difficult topics
        elif energy_level >= 4 and mood_score >= 4:
            for task in today_schedule:
                if task['difficulty'] == 'easy':
                    # Suggest adding a challenging topic
                    hard_topic = conn.execute('''
                        SELECT st.* FROM syllabus_topics st
                        JOIN subjects s ON st.subject_id = s.subject_id
                        WHERE s.user_id = ? AND st.difficulty = 'hard' 
                        AND st.is_completed = 0
                        LIMIT 1
                    ''', (user_id,)).fetchone()
                    
                    if hard_topic:
                        adaptations.append({
                            'action': 'add_challenge',
                            'suggestion': {
                                'topic_id': hard_topic['topic_id'],
                                'topic_name': hard_topic['topic_name'],
                                'difficulty': 'hard'
                            },
                            'reason': 'High energy day - good time for challenging topics'
                        })
                        break
        
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

@app.route('/setup_profile')
@login_required
def setup_profile():
    return render_template('setup_profile.html')

@app.route('/save_profile_setup', methods=['POST'])
@login_required
def save_profile_setup():
    user_id = session['user_id']
    
    # Save user preferences
    daily_hours = request.form.get('daily_hours', 4)
    preferred_slot = request.form.get('preferred_slot', 'morning')
    break_length = request.form.get('break_length', 15)
    study_method = request.form.get('study_method', 'learning')
    
    conn = get_db()
    conn.execute('''
        UPDATE user_schedule_preferences 
        SET daily_study_hours = ?, preferred_study_slot = ?, 
            break_length_minutes = ?
        WHERE user_id = ?
    ''', (daily_hours, preferred_slot, break_length, user_id))
    
    # Save study method preference
    conn.execute('''
        INSERT INTO study_methods_preferences (user_id, preferred_method)
        VALUES (?, ?)
    ''', (user_id, study_method))
    
    conn.commit()
    conn.close()
    
    return redirect('/dashboard')


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
            session['email'] = user['email']
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
    
        # Get subjects with progress
    subjects = conn.execute('''
        SELECT s.*, 
               COUNT(st.topic_id) as total_topics,
               SUM(CASE WHEN st.is_completed = 1 THEN 1 ELSE 0 END) as completed_topics
        FROM subjects s
        LEFT JOIN syllabus_topics st ON s.subject_id = st.subject_id
        WHERE s.user_id = ?
        GROUP BY s.subject_id
        ORDER BY s.exam_date
    ''', (user_id,)).fetchall()
    
    # Calculate overall progress
    total_topics = sum(s['total_topics'] or 0 for s in subjects)
    completed_topics = sum(s['completed_topics'] or 0 for s in subjects)
    overall_progress = (completed_topics / total_topics * 100) if total_topics > 0 else 0
    

    
 # Get today's schedule
    today = datetime.now().strftime('%Y-%m-%d')
    today_schedule = conn.execute('''
        SELECT ss.*, s.subject_name, st.topic_name, st.difficulty
        FROM study_schedules ss
        LEFT JOIN subjects s ON ss.subject_id = s.subject_id
        LEFT JOIN syllabus_topics st ON ss.topic_id = st.topic_id
        WHERE ss.user_id = ? AND ss.scheduled_date = ?
        ORDER BY ss.start_time
    ''', (user_id, today)).fetchall()
    
    # Get upcoming deadlines (next 7 days)
    upcoming_deadlines = conn.execute('''
        SELECT subject_name, exam_date 
        FROM subjects 
        WHERE user_id = ? AND exam_date IS NOT NULL 
        AND DATE(exam_date) BETWEEN DATE('now') AND DATE('now', '+7 days')
        ORDER BY exam_date
    ''', (user_id,)).fetchall()
    
    # Get recent mood
    recent_mood = conn.execute('''
        SELECT * FROM mood_tracking 
        WHERE user_id = ? 
        ORDER BY tracking_date DESC 
        LIMIT 1
    ''', (user_id,)).fetchone()
    
    conn.close()
    
    return render_template('dashboard.html',
                         user=user,
                         subjects=subjects,
                         overall_progress=round(overall_progress, 1),
                         today_schedule=today_schedule,
                         upcoming_deadlines=upcoming_deadlines,
                         recent_mood=recent_mood)


@app.route('/subject_management')
@login_required
def subject_management():
    user_id = session['user_id']
    
    conn = get_db()
    subjects = conn.execute(
        'SELECT * FROM subjects WHERE user_id = ? ORDER BY exam_date',
        (user_id,)
    ).fetchall()
    conn.close()
    
    return render_template('subject_management.html', subjects=subjects)

@app.route('/add_subject', methods=['POST'])
@login_required
def add_subject():
    user_id = session['user_id']
    
    subject_name = request.form['subject_name']
    class_degree = request.form.get('class_degree', '')
    exam_date = request.form.get('exam_date', None)
    self_rating = request.form.get('self_rating', 5)
    target_marks = request.form.get('target_marks', 'average')
    weak_areas = request.form.get('weak_areas', '')
    strong_areas = request.form.get('strong_areas', '')
    
    conn = get_db()
    conn.execute('''
        INSERT INTO subjects 
        (user_id, subject_name, exam_date, self_rating, current_level, weak_areas, strong_areas)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, subject_name, exam_date, self_rating, class_degree, weak_areas, strong_areas))
    
    # Also add as a goal
    if exam_date:
        conn.execute('''
            INSERT INTO user_goals (user_id, goal_type, description, target_date, target_marks)
            VALUES (?, 'exam-specific', ?, ?, ?)
        ''', (user_id, f"Prepare for {subject_name} exam", exam_date, target_marks))
    
    conn.commit()
    conn.close()
    
    return redirect('/subject_management')

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
    
    # Get next sequence order
    max_order = conn.execute(
        'SELECT MAX(sequence_order) as max_order FROM syllabus_topics WHERE subject_id = ?',
        (subject_id,)
    ).fetchone()
    next_order = (max_order['max_order'] or 0) + 1
    
    # Use AI to estimate time
    subject = conn.execute(
        'SELECT self_rating FROM subjects WHERE subject_id = ?',
        (subject_id,)
    ).fetchone()
    
    ai_planner = AdvancedStudyPlannerAI()
    estimated_time = ai_planner.estimate_study_time(
        topic_name, difficulty, subject['self_rating'] if subject else 5
    )
    
    conn.execute('''
        INSERT INTO syllabus_topics 
        (subject_id, topic_name, difficulty, priority_level, sequence_order, estimated_time_hours)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (subject_id, topic_name, difficulty, priority, next_order, estimated_time))
    
    conn.commit()
    conn.close()
    
    return redirect('/subject_management')


@app.route('/upload_syllabus', methods=['POST'])
@login_required
def upload_syllabus():
    user_id = session['user_id']
    subject_id = request.form['subject_id']
    
    if 'syllabus_file' not in request.files:
        return "No file uploaded"
    
    file = request.files['syllabus_file']
    
    if file.filename == '':
        return "No file selected"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{user_id}_{subject_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # In a real application, you would use OCR/PDF parsing here
        # For now, we'll simulate extraction
        simulated_topics = [
            "Introduction to Subject",
            "Basic Concepts",
            "Advanced Topics",
            "Applications",
            "Revision and Practice"
        ]
        
        conn = get_db()
        for i, topic in enumerate(simulated_topics):
            conn.execute('''
                INSERT INTO syllabus_topics 
                (subject_id, topic_name, difficulty, priority_level, sequence_order)
                VALUES (?, ?, 'medium', 2, ?)
            ''', (subject_id, topic, i+1))
        
        conn.commit()
        conn.close()
        
        return redirect('/subject_management')
    
    return "Invalid file type"


@app.route('/time_availability')
@login_required
def time_availability():
    user_id = session['user_id']
    
    conn = get_db()
    preferences = conn.execute(
        'SELECT * FROM user_schedule_preferences WHERE user_id = ?',
        (user_id,)
    ).fetchone()
    
    constraints = conn.execute(
        'SELECT * FROM user_constraints WHERE user_id = ? ORDER BY constraint_date',
        (user_id,)
    ).fetchall()
    
    conn.close()
    
    return render_template('time_availability.html', 
                         preferences=preferences, 
                         constraints=constraints)

@app.route('/save_time_preferences', methods=['POST'])
@login_required
def save_time_preferences():
    user_id = session['user_id']
    
    daily_hours = request.form.get('daily_hours', 4)
    preferred_slot = request.form.get('preferred_slot', 'morning')
    break_length = request.form.get('break_length', 15)
    max_daily_load = request.form.get('max_daily_load', 8)
    sleep_time = request.form.get('sleep_time', '23:00')
    wake_up_time = request.form.get('wake_up_time', '07:00')
    
    conn = get_db()
    conn.execute('''
        UPDATE user_schedule_preferences 
        SET daily_study_hours = ?, preferred_study_slot = ?, break_length_minutes = ?,
            max_daily_load_hours = ?, sleep_time = ?, wake_up_time = ?
        WHERE user_id = ?
    ''', (daily_hours, preferred_slot, break_length, max_daily_load, sleep_time, wake_up_time, user_id))
    
    conn.commit()
    conn.close()
    
    return redirect('/time_availability')

@app.route('/add_constraint', methods=['POST'])
@login_required
def add_constraint():
    user_id = session['user_id']
    
    constraint_date = request.form['constraint_date']
    constraint_type = request.form['constraint_type']
    description = request.form.get('description', '')
    start_time = request.form.get('start_time', None)
    end_time = request.form.get('end_time', None)
    
    conn = get_db()
    conn.execute('''
        INSERT INTO user_constraints 
        (user_id, constraint_date, constraint_type, description, start_time, end_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, constraint_date, constraint_type, description, start_time, end_time))
    
    conn.commit()
    conn.close()
    
    return redirect('/time_availability')

@app.route('/goals')
@login_required
def goals():
    user_id = session['user_id']
    
    conn = get_db()
    user_goals = conn.execute(
        'SELECT * FROM user_goals WHERE user_id = ? ORDER BY target_date',
        (user_id,)
    ).fetchall()
    
    conn.close()
    
    return render_template('goals.html', goals=user_goals)

@app.route('/add_goal', methods=['POST'])
@login_required
def add_goal():
    user_id = session['user_id']
    
    goal_type = request.form['goal_type']
    description = request.form['description']
    target_date = request.form.get('target_date', None)
    target_marks = request.form.get('target_marks', 'average')
    
    conn = get_db()
    conn.execute('''
        INSERT INTO user_goals (user_id, goal_type, description, target_date, target_marks)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, goal_type, description, target_date, target_marks))
    
    conn.commit()
    conn.close()
    
    return redirect('/goals')

@app.route('/ai_schedule_generator')
@login_required
def ai_schedule_generator():
    return render_template('ai_schedule_generator.html')

@app.route('/generate_ai_schedule', methods=['POST'])
@login_required
def generate_ai_schedule():
    user_id = session['user_id']
    
    try:
        # Get parameters
        start_date = request.form.get('start_date', datetime.now().strftime('%Y-%m-%d'))
        days_ahead = int(request.form.get('days_ahead', 30))
        schedule_type = request.form.get('schedule_type', 'balanced')
        
        # Generate AI schedule
        ai_planner = AdvancedStudyPlannerAI()
        schedule = ai_planner.create_smart_schedule(user_id, start_date, days_ahead)
        
        if 'error' in schedule:
            return render_template('ai_schedule_result.html', error=schedule['error'])
        
        # Save to database
        conn = get_db()
        
        # Clear existing schedule for the period
        end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        conn.execute('''
            DELETE FROM study_schedules 
            WHERE user_id = ? AND scheduled_date BETWEEN ? AND ?
        ''', (user_id, start_date, end_date))
        
        # Insert new schedule
        for day in schedule:
            for slot in day['slots']:
                conn.execute('''
                    INSERT INTO study_schedules 
                    (user_id, subject_id, topic_id, scheduled_date, start_time, end_time, study_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, slot['subject_id'], slot['topic_id'], 
                      day['date'], slot['start_time'], slot['end_time'], slot['study_method']))
        
        conn.commit()
        
        # Get detailed schedule for display
        detailed_schedule = []
        for day in schedule:
            day_details = {
                'date': day['date'],
                'day_name': day['day_name'],
                'total_hours': day['total_hours'],
                'slots': []
            }
            
            for slot in day['slots']:
                # Get subject and topic names
                subject = conn.execute(
                    'SELECT subject_name FROM subjects WHERE subject_id = ?',
                    (slot['subject_id'],)
                ).fetchone()
                
                topic = conn.execute(
                    'SELECT topic_name FROM syllabus_topics WHERE topic_id = ?',
                    (slot['topic_id'],)
                ).fetchone()
                
                day_details['slots'].append({
                    'time': f"{slot['start_time']} - {slot['end_time']}",
                    'subject_name': subject['subject_name'] if subject else 'Unknown',
                    'topic_name': topic['topic_name'] if topic else 'Unknown',
                    'difficulty': slot['difficulty'],
                    'study_method': slot['study_method'],
                    'duration': slot['duration']
                })
            
            detailed_schedule.append(day_details)
        
        conn.close()
        
        # Generate topic breakdown
        topic_breakdown = []
        for day in schedule:
            for slot in day['slots']:
                topic_breakdown.append({
                    'topic_name': slot['topic_name'],
                    'estimated_time': slot['duration'],
                    'priority': 'High' if slot['difficulty'] == 'hard' else 'Medium',
                    'scheduled_date': day['date']
                })
        
        # Remove duplicates
        unique_breakdown = []
        seen = set()
        for item in topic_breakdown:
            identifier = item['topic_name']
            if identifier not in seen:
                seen.add(identifier)
                unique_breakdown.append(item)
        
        return render_template('ai_schedule_result.html',
                             schedule=detailed_schedule,
                             topic_breakdown=unique_breakdown,
                             days_ahead=days_ahead,
                             schedule_type=schedule_type)
    
    except Exception as e:
        return render_template('ai_schedule_result.html', 
                             error=f"Error generating schedule: {str(e)}")
    
@app.route('/calendar_view')
@login_required
def calendar_view():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get schedule for current month
    today = datetime.now()
    month_start = today.replace(day=1).strftime('%Y-%m-%d')
    next_month = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
    month_end = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
    
    monthly_schedule = conn.execute('''
        SELECT ss.*, s.subject_name, st.topic_name, st.difficulty
        FROM study_schedules ss
        LEFT JOIN subjects s ON ss.subject_id = s.subject_id
        LEFT JOIN syllabus_topics st ON ss.topic_id = st.topic_id
        WHERE ss.user_id = ? AND ss.scheduled_date BETWEEN ? AND ?
        ORDER BY ss.scheduled_date, ss.start_time
    ''', (user_id, month_start, month_end)).fetchall()
    
    # Group by date
    schedule_by_date = {}
    for item in monthly_schedule:
        date_str = item['scheduled_date']
        if date_str not in schedule_by_date:
            schedule_by_date[date_str] = []
        schedule_by_date[date_str].append({
            'subject': item['subject_name'],
            'topic': item['topic_name'],
            'time': f"{item['start_time']} - {item['end_time']}",
            'difficulty': item['difficulty'],
            'is_completed': bool(item['is_completed'])
        })
    
    # Calculate statistics
    total_sessions = len(monthly_schedule)
    completed_sessions = sum(1 for item in monthly_schedule if item['is_completed'])
    
    # Generate calendar dates for current month
    calendar_dates = []
    current = datetime.strptime(month_start, '%Y-%m-%d')
    
    # Find first Sunday
    while current.weekday() != 6:  # 6 = Sunday
        current -= timedelta(days=1)
    
    # Generate 42 days (6 weeks) for calendar
    for i in range(42):
        date_str = current.strftime('%Y-%m-%d')
        day_schedule = schedule_by_date.get(date_str, [])
        calendar_dates.append({
            'date': date_str,
            'day': current.day,
            'is_today': date_str == today.strftime('%Y-%m-%d'),
            'has_schedule': len(day_schedule) > 0,
            'sessions': day_schedule
        })
        current += timedelta(days=1)
    
    # Calculate study days (unique days with schedule in current month)
    study_days = sum(1 for date_str, sessions in schedule_by_date.items() 
                    if month_start <= date_str <= month_end and sessions)
    
    conn.close()
    
    return render_template('calendar_view.html', 
                         calendar_dates=calendar_dates,
                         schedule_by_date=schedule_by_date,
                         current_month=today.strftime('%B %Y'),
                         total_sessions=total_sessions,
                         completed_sessions=completed_sessions,
                         study_days=study_days)




# ========== progress_tracking ROUTE ==========

@app.route('/progress_tracking')
@login_required
def progress_tracking():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get progress data - FIXED: removed schedule_id reference
    progress_data = conn.execute('''
        SELECT date, SUM(time_spent_minutes) as total_minutes, COUNT(*) as sessions
        FROM progress_tracking 
        WHERE user_id = ? 
        GROUP BY date 
        ORDER BY date DESC 
        LIMIT 30
    ''', (user_id,)).fetchall()
    
    # Get subject-wise progress
    subject_progress = conn.execute('''
        SELECT s.subject_name, 
               COUNT(st.topic_id) as total_topics,
               SUM(CASE WHEN st.is_completed = 1 THEN 1 ELSE 0 END) as completed_topics,
               (SUM(CASE WHEN st.is_completed = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(st.topic_id)) as completion_rate
        FROM subjects s
        LEFT JOIN syllabus_topics st ON s.subject_id = st.subject_id
        WHERE s.user_id = ?
        GROUP BY s.subject_id
        HAVING COUNT(st.topic_id) > 0
    ''', (user_id,)).fetchall()
    
    # Get streak - SIMPLIFIED version without complex CTE
    # First get all dates with completed study sessions in last 30 days
    completed_dates = conn.execute('''
        SELECT DISTINCT scheduled_date 
        FROM study_schedules 
        WHERE user_id = ? 
        AND is_completed = 1
        AND scheduled_date >= DATE('now', '-30 days')
        ORDER BY scheduled_date DESC
    ''', (user_id,)).fetchall()
    
    # Calculate streak manually
    current_streak = 0
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # Check if studied today
    studied_today = conn.execute('''
        SELECT COUNT(*) as count 
        FROM study_schedules 
        WHERE user_id = ? 
        AND scheduled_date = DATE('now')
        AND is_completed = 1
    ''', (user_id,)).fetchone()['count'] > 0
    
    if studied_today:
        current_streak += 1
        
        # Check consecutive days
        check_date = yesterday
        while True:
            studied = conn.execute('''
                SELECT COUNT(*) as count 
                FROM study_schedules 
                WHERE user_id = ? 
                AND scheduled_date = ?
                AND is_completed = 1
            ''', (user_id, check_date.strftime('%Y-%m-%d'))).fetchone()['count'] > 0
            
            if studied:
                current_streak += 1
                check_date -= timedelta(days=1)
            else:
                break
    
    conn.close()
    
    # Calculate statistics for template
    total_minutes = sum(row['total_minutes'] or 0 for row in progress_data)
    avg_daily = total_minutes / 30 if total_minutes > 0 else 0
    
    return render_template('progress_tracking.html',
                         progress_data=progress_data,
                         subject_progress=subject_progress,
                         streak=current_streak,
                         total_minutes=total_minutes,
                         avg_daily=avg_daily)



@app.route('/log_study_session', methods=['POST'])
@login_required
def log_study_session():
    user_id = session['user_id']
    
    schedule_id = request.form.get('schedule_id')
    time_spent = request.form.get('time_spent', 0)
    topics_covered = request.form.get('topics_covered', '')
    self_assessment = request.form.get('self_assessment', 3)
    
    conn = get_db()
    
    # Get schedule details
    schedule = conn.execute(
        'SELECT * FROM study_schedules WHERE schedule_id = ?',
        (schedule_id,)
    ).fetchone()
    
    if schedule:
        # Update schedule completion
        conn.execute(
            'UPDATE study_schedules SET is_completed = 1, completion_percentage = 100 WHERE schedule_id = ?',
            (schedule_id,)
        )
        
        # Log progress
        conn.execute('''
            INSERT INTO progress_tracking 
            (user_id, subject_id, date, time_spent_minutes, topics_covered, self_assessment_score)
            VALUES (?, ?, DATE('now'), ?, ?, ?)
        ''', (user_id, schedule['subject_id'], time_spent, topics_covered, self_assessment))
        
        conn.commit()
    
    conn.close()
    
    return jsonify({'status': 'success', 'message': 'Study session logged!'})

# Add to your existing routes

@app.route('/ask_ai_tutor', methods=['POST'])
@login_required
def ask_ai_tutor():
    user_id = session['user_id']
    data = request.json
    question = data.get('question', '')
    topic_id = data.get('topic_id')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    ai_tutor = AITutor(user_id)
    response = ai_tutor.ask_ai_tutor(question, topic_id)
    
    return jsonify(response)

@app.route('/generate_revision_strategy', methods=['POST'])
@login_required
def generate_revision_strategy():
    user_id = session['user_id']
    data = request.json
    topic_id = data.get('topic_id')
    days = data.get('days', 7)
    
    if not topic_id:
        return jsonify({'error': 'Topic ID is required'}), 400
    
    ai_tutor = AITutor(user_id)
    strategy = ai_tutor.generate_smart_revision_strategy(topic_id)
    
    return jsonify(strategy)

@app.route('/generate_flashcards', methods=['POST'])
@login_required
def generate_flashcards():
    user_id = session['user_id']
    data = request.json
    topic_id = data.get('topic_id')
    count = data.get('count', 10)
    difficulty = data.get('difficulty', 'mixed')
    
    if not topic_id:
        return jsonify({'error': 'Topic ID is required'}), 400
    
    ai_tutor = AITutor(user_id)
    flashcards = ai_tutor.generate_ai_flashcards(topic_id, count)
    
    return jsonify({'flashcards': flashcards})

@app.route('/generate_practice_questions', methods=['POST'])
@login_required
def generate_practice_questions():
    user_id = session['user_id']
    data = request.json
    topic_id = data.get('topic_id')
    count = data.get('count', 5)
    difficulty = data.get('difficulty', 'mixed')
    
    if not topic_id:
        return jsonify({'error': 'Topic ID is required'}), 400
    
    ai_tutor = AITutor(user_id)
    questions = ai_tutor.generate_practice_questions(topic_id, count, difficulty)
    
    return jsonify({'questions': questions})

@app.route('/get_next_topic_recommendation', methods=['POST'])
@login_required
def get_next_topic_recommendation():
    user_id = session['user_id']
    
    ai_tutor = AITutor(user_id)
    recommendation = ai_tutor.get_next_topic_recommendation()
    
    return jsonify(recommendation)

# Update the ai_tutoring route to pass topics
@app.route('/ai_tutoring')
@login_required
def ai_tutoring():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get subjects and topics
    subjects = conn.execute(
        'SELECT * FROM subjects WHERE user_id = ? ORDER BY subject_name',
        (user_id,)
    ).fetchall()
    
    topics = conn.execute('''
        SELECT st.*, s.subject_name 
        FROM syllabus_topics st
        JOIN subjects s ON st.subject_id = s.subject_id
        WHERE s.user_id = ? AND st.is_completed = 0
        ORDER BY st.topic_name
    ''', (user_id,)).fetchall()
    
    # Get recent AI content
    recent_content = conn.execute('''
        SELECT * FROM ai_content 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 5
    ''', (user_id,)).fetchall()
    
    conn.close()
    
    return render_template('ai_tutoring.html',
                         subjects=subjects,
                         topics=topics,
                         recent_content=recent_content)


@app.route('/mood_tracker')
@login_required
def mood_tracker():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get mood history
    mood_history = conn.execute('''
        SELECT * FROM mood_tracking 
        WHERE user_id = ? 
        ORDER BY tracking_date DESC 
        LIMIT 14
    ''', (user_id,)).fetchall()
    
    # Get today's mood if exists
    today = datetime.now().strftime('%Y-%m-%d')
    today_mood = conn.execute(
        'SELECT * FROM mood_tracking WHERE user_id = ? AND tracking_date = ?',
        (user_id, today)
    ).fetchone()
    
    conn.close()
    
    return render_template('mood_tracker.html',
                         mood_history=mood_history,
                         today_mood=today_mood)

@app.route('/log_mood_daily', methods=['POST'])
@login_required
def log_mood_daily():
    user_id = session['user_id']
    
    mood_score = request.form['mood_score']
    energy_level = request.form['energy_level']
    stress_level = request.form['stress_level']
    notes = request.form.get('notes', '')
    
    conn = get_db()
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if already logged today
    existing = conn.execute(
        'SELECT * FROM mood_tracking WHERE user_id = ? AND tracking_date = ?',
        (user_id, today)
    ).fetchone()
    
    if existing:
        conn.execute('''
            UPDATE mood_tracking 
            SET mood_score = ?, energy_level = ?, stress_level = ?, notes = ?
            WHERE mood_id = ?
        ''', (mood_score, energy_level, stress_level, notes, existing['mood_id']))
    else:
        conn.execute('''
            INSERT INTO mood_tracking 
            (user_id, mood_score, energy_level, stress_level, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, mood_score, energy_level, stress_level, notes))
    
    # Get AI adaptations
    ai_planner = AdvancedStudyPlannerAI()
    adaptations = ai_planner.adapt_schedule_for_mood(user_id, {
        'mood_score': int(mood_score),
        'energy_level': int(energy_level),
        'stress_level': int(stress_level)
    })
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'status': 'success',
        'message': 'Mood logged successfully!',
        'adaptations': adaptations,
        'suggestions': get_wellbeing_suggestions(
            int(mood_score), int(energy_level), int(stress_level)
            )
    })

def get_wellbeing_suggestions(mood, energy, stress):
    """Get wellbeing suggestions based on mood"""
    suggestions = []
    
    if stress >= 4:
        suggestions.append("🧘 Take a 5-minute meditation break")
        suggestions.append("💧 Drink water and stretch")
        suggestions.append("🌳 Take a short walk outside")
    
    if energy <= 2:
        suggestions.append("🍎 Have a healthy snack")
        suggestions.append("☕ Take a short break")
        suggestions.append("🎵 Listen to some energizing music")
    
    if mood <= 2:
        suggestions.append("📞 Talk to a friend for 5 minutes")
        suggestions.append("🎯 Break tasks into smaller chunks")
        suggestions.append("📝 Write down what's bothering you")
    
    return suggestions[:3]  # Return top 3 suggestions

@app.template_filter('to_datetime')
def to_datetime(value):
    """Convert string to datetime for template use"""
    if isinstance(value, str):
        try:
            return datetime.strptime(value, '%Y-%m-%d')
        except:
            return datetime.now()
    return value

@app.route('/notifications')
@login_required
def notifications():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get upcoming notifications
    upcoming_notifications = conn.execute('''
        SELECT * FROM notifications 
        WHERE user_id = ? AND is_sent = 0 AND scheduled_time > DATETIME('now')
        ORDER BY scheduled_time
        LIMIT 10
    ''', (user_id,)).fetchall()
    
    # Get past notifications
    past_notifications = conn.execute('''
        SELECT * FROM notifications 
        WHERE user_id = ? AND is_sent = 1
        ORDER BY sent_time DESC
        LIMIT 20
    ''', (user_id,)).fetchall()
    
    conn.close()
    
    return render_template('notifications.html',
                         upcoming=upcoming_notifications,
                         past=past_notifications)

@app.route('/settings')
@login_required
def settings():
    user_id = session['user_id']
    
    conn = get_db()
    profile = conn.execute(
        'SELECT * FROM profiles WHERE user_id = ?',
        (user_id,)
    ).fetchone()
    conn.close()
    
    return render_template('settings.html', profile=profile)

@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    user_id = session['user_id']
    
    theme = request.form.get('theme', 'light')
    font_size = request.form.get('font_size', 14)
    color_mode = request.form.get('color_mode', 'default')
    notifications_enabled = request.form.get('notifications', 'on') == 'on'
    
    conn = get_db()
    conn.execute('''
        UPDATE profiles 
        SET theme_preference = ?, font_size = ?, color_mode = ?
        WHERE user_id = ?
    ''', (theme, font_size, color_mode, user_id))
    
    conn.commit()
    conn.close()
    
    return redirect('/settings')

@app.route('/export_schedule')
@login_required
def export_schedule():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get schedule
    schedule = conn.execute('''
        SELECT ss.scheduled_date, ss.start_time, ss.end_time, 
               s.subject_name, st.topic_name
        FROM study_schedules ss
        LEFT JOIN subjects s ON ss.subject_id = s.subject_id
        LEFT JOIN syllabus_topics st ON ss.topic_id = st.topic_id
        WHERE ss.user_id = ? AND ss.scheduled_date >= DATE('now')
        ORDER BY ss.scheduled_date, ss.start_time
    ''', (user_id,)).fetchall()
    
    conn.close()
    
    # Generate ICS file content
    ics_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//AI Study Planner//EN\n"
    
    for item in schedule:
        # Format dates for ICS
        date_str = item['scheduled_date'].replace('-', '')
        start_time = item['start_time'].replace(':', '') if item['start_time'] else '090000'
        end_time = item['end_time'].replace(':', '') if item['end_time'] else '110000'
        
        ics_content += f"""BEGIN:VEVENT
UID:{uuid.uuid4()}
DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{date_str}T{start_time}00
DTEND:{date_str}T{end_time}00
SUMMARY:Study: {item['subject_name']} - {item['topic_name']}
DESCRIPTION:Study session for {item['subject_name']}
END:VEVENT
"""
    
    ics_content += "END:VCALENDAR"
    
    # Create response
    from io import BytesIO
    bio = BytesIO()
    bio.write(ics_content.encode('utf-8'))
    bio.seek(0)
    
    return send_file(
        bio,
        as_attachment=True,
        download_name='study_schedule.ics',
        mimetype='text/calendar'
    )

@app.route('/update_session_progress', methods=['POST'])
@login_required
def update_session_progress():
    user_id = session['user_id']
    schedule_id = request.form['schedule_id']
    completion_status = request.form['completion_status']
    time_spent = request.form.get('time_spent', 0)
    notes = request.form.get('notes', '')
    
    conn = get_db()
    
    try:
        # Update schedule completion
        if completion_status == 'completed':
            is_completed = 1
            completion_percentage = 100
        elif completion_status == 'partial':
            is_completed = 0
            completion_percentage = 50
        else:  # skipped
            is_completed = 0
            completion_percentage = 0
        
        conn.execute('''
            UPDATE study_schedules 
            SET is_completed = ?, completion_percentage = ?
            WHERE schedule_id = ? AND user_id = ?
        ''', (is_completed, completion_percentage, schedule_id, user_id))
        
        # Log in progress tracking
        if time_spent and int(time_spent) > 0:
            # Check if already logged today
            existing = conn.execute('''
                SELECT * FROM progress_tracking 
                WHERE user_id = ? AND date = DATE('now') AND schedule_id = ?
            ''', (user_id, schedule_id)).fetchone()
            
            if existing:
                conn.execute('''
                    UPDATE progress_tracking 
                    SET time_spent_minutes = ?, notes = ?
                    WHERE progress_id = ?
                ''', (time_spent, notes, existing['progress_id']))
            else:
                # Get subject_id from schedule
                schedule = conn.execute('''
                    SELECT subject_id FROM study_schedules WHERE schedule_id = ?
                ''', (schedule_id,)).fetchone()
                
                if schedule:
                    conn.execute('''
                        INSERT INTO progress_tracking 
                        (user_id, schedule_id, subject_id, date, time_spent_minutes, notes)
                        VALUES (?, ?, ?, DATE('now'), ?, ?)
                    ''', (user_id, schedule_id, schedule['subject_id'], time_spent, notes))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Progress updated successfully!'
        })
        
    except Exception as e:
        conn.close()
        return jsonify({
            'status': 'error',
            'message': f'Error updating progress: {str(e)}'
        })

@app.route('/get_progress_analytics')
@login_required
def get_progress_analytics():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get study time distribution by subject
    subject_distribution = conn.execute('''
        SELECT 
            s.subject_name,
            SUM(strftime('%s', ss.end_time) - strftime('%s', ss.start_time)) / 3600.0 as total_hours,
            ROUND(SUM(strftime('%s', ss.end_time) - strftime('%s', ss.start_time)) * 100.0 / 
                  (SELECT SUM(strftime('%s', ss2.end_time) - strftime('%s', ss2.start_time)) 
                   FROM study_schedules ss2 WHERE ss2.user_id = ?), 1) as percentage
        FROM study_schedules ss
        JOIN subjects s ON ss.subject_id = s.subject_id
        WHERE ss.user_id = ? AND ss.scheduled_date >= DATE('now', '-30 days')
        GROUP BY s.subject_id
        HAVING total_hours > 0
        ORDER BY total_hours DESC
    ''', (user_id, user_id)).fetchall()
    
    # Get progress over time (weekly)
    weekly_progress = conn.execute('''
        SELECT 
            strftime('%Y-%W', ss.scheduled_date) as week,
            MIN(ss.scheduled_date) as week_start,
            MAX(ss.scheduled_date) as week_end,
            COUNT(*) as total_sessions,
            SUM(CASE WHEN ss.is_completed = 1 THEN 1 ELSE 0 END) as completed_sessions,
            SUM(strftime('%s', ss.end_time) - strftime('%s', ss.start_time)) / 3600.0 as total_hours
        FROM study_schedules ss
        WHERE ss.user_id = ? 
        GROUP BY strftime('%Y-%W', ss.scheduled_date)
        ORDER BY week_start DESC
        LIMIT 8
    ''', (user_id,)).fetchall()
    
    # Get performance metrics
    performance_metrics = conn.execute('''
        SELECT 
            'Session Completion Rate' as metric,
            ROUND(AVG(CASE WHEN ss.is_completed = 1 THEN 100.0 ELSE 0 END), 1) as value,
            '%' as unit
        FROM study_schedules ss
        WHERE ss.user_id = ? AND ss.scheduled_date >= DATE('now', '-30 days')
        UNION ALL
        SELECT 
            'Avg Study Time per Day' as metric,
            ROUND(AVG(strftime('%s', ss.end_time) - strftime('%s', ss.start_time)) / 3600.0, 1) as value,
            'hours' as unit
        FROM study_schedules ss
        WHERE ss.user_id = ? AND ss.scheduled_date >= DATE('now', '-7 days')
        UNION ALL
        SELECT 
            'Topics Completed This Week' as metric,
            COUNT(*) as value,
            'topics' as unit
        FROM syllabus_topics st
        JOIN subjects s ON st.subject_id = s.subject_id
        WHERE s.user_id = ? AND st.completed_date >= DATE('now', '-7 days')
        UNION ALL
        SELECT 
            'Current Study Streak' as metric,
            (SELECT COUNT(*) FROM (
                WITH RECURSIVE dates(date) AS (
                    SELECT DATE('now')
                    UNION ALL
                    SELECT DATE(date, '-1 day')
                    FROM dates
                    WHERE date > DATE('now', '-60 days')
                )
                SELECT 1 FROM dates d
                WHERE EXISTS (
                    SELECT 1 FROM study_schedules ss
                    WHERE ss.user_id = ? 
                    AND ss.scheduled_date = d.date
                    AND ss.is_completed = 1
                )
                AND d.date <= DATE('now')
                ORDER BY d.date DESC
            )) as value,
            'days' as unit
    ''', (user_id, user_id, user_id, user_id)).fetchall()
    
    conn.close()
    
    return jsonify({
        'subject_distribution': [dict(row) for row in subject_distribution],
        'weekly_progress': [dict(row) for row in weekly_progress],
        'performance_metrics': [dict(row) for row in performance_metrics]
    })

@app.route('/export_progress_report')
@login_required
def export_progress_report():
    user_id = session['user_id']
    
    conn = get_db()
    
    # Get user info
    user = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,)).fetchone()
    
    # Get comprehensive progress data
    progress_data = conn.execute('''
        SELECT 
            s.subject_name,
            COUNT(st.topic_id) as total_topics,
            SUM(CASE WHEN st.is_completed = 1 THEN 1 ELSE 0 END) as completed_topics,
            ROUND((SUM(CASE WHEN st.is_completed = 1 THEN 1 ELSE 0 END) * 100.0 / 
                  NULLIF(COUNT(st.topic_id), 0)), 1) as completion_percentage,
            s.exam_date,
            s.self_rating
        FROM subjects s
        LEFT JOIN syllabus_topics st ON s.subject_id = st.subject_id
        WHERE s.user_id = ?
        GROUP BY s.subject_id
        ORDER BY completion_percentage DESC
    ''', (user_id,)).fetchall()
    
    # Get study time summary
    time_summary = conn.execute('''
        SELECT 
            strftime('%Y-%m', ss.scheduled_date) as month,
            COUNT(*) as total_sessions,
            SUM(CASE WHEN ss.is_completed = 1 THEN 1 ELSE 0 END) as completed_sessions,
            SUM(strftime('%s', ss.end_time) - strftime('%s', ss.start_time)) / 3600.0 as total_hours
        FROM study_schedules ss
        WHERE ss.user_id = ?
        GROUP BY strftime('%Y-%m', ss.scheduled_date)
        ORDER BY month DESC
        LIMIT 3
    ''', (user_id,)).fetchall()
    
    conn.close()
    
    # Generate report text
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("AI STUDY PLANNER - PROGRESS REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Student: {user['username']}")
    report_lines.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")
    report_lines.append("SUBJECT PROGRESS SUMMARY:")
    report_lines.append("-" * 40)
    
    total_completion = 0
    total_subjects = len(progress_data)
    
    for subject in progress_data:
        report_lines.append(f"{subject['subject_name']}:")
        report_lines.append(f"  Topics: {subject['completed_topics']}/{subject['total_topics']} "
                          f"({subject['completion_percentage']}%)")
        if subject['exam_date']:
            days_left = (datetime.strptime(subject['exam_date'], '%Y-%m-%d').date() - 
                        datetime.now().date()).days
            report_lines.append(f"  Exam: {subject['exam_date']} (in {days_left} days)")
        report_lines.append("")
        total_completion += subject['completion_percentage'] or 0
    
    avg_completion = total_completion / total_subjects if total_subjects > 0 else 0
    
    report_lines.append("STUDY TIME SUMMARY:")
    report_lines.append("-" * 40)
    for month in time_summary:
        report_lines.append(f"{month['month']}:")
        report_lines.append(f"  Sessions: {month['completed_sessions']}/{month['total_sessions']} "
                          f"({(month['completed_sessions']/month['total_sessions']*100 if month['total_sessions']>0 else 0):.1f}%)")
        report_lines.append(f"  Total Hours: {month['total_hours']:.1f}")
        report_lines.append("")
    
    report_lines.append("OVERALL STATISTICS:")
    report_lines.append("-" * 40)
    report_lines.append(f"Average Completion Rate: {avg_completion:.1f}%")
    report_lines.append(f"Total Subjects: {total_subjects}")
    report_lines.append("")
    report_lines.append("RECOMMENDATIONS:")
    report_lines.append("-" * 40)
    
    # Add recommendations
    if avg_completion < 50:
        report_lines.append("1. Focus on completing at least 50% of all topics")
    if any(s['completion_percentage'] < 30 for s in progress_data):
        report_lines.append("2. Identify and prioritize subjects with lowest completion")
    
    report_lines.append("3. Maintain consistent daily study schedule")
    report_lines.append("4. Review completed topics regularly")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    report_lines.append("Generated by AI Study Planner")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    # Create downloadable file
    from io import BytesIO
    bio = BytesIO()
    bio.write(report_text.encode('utf-8'))
    bio.seek(0)
    
    return send_file(
        bio,
        as_attachment=True,
        download_name=f'progress_report_{datetime.now().strftime("%Y%m%d")}.txt',
        mimetype='text/plain'
    )



@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ========== MAIN ==========

if __name__ == '__main__':
    # Initialize database
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"⚠️ Database initialization note: {e}")
    
    print("🚀 AI Study Planner is running!")
    print("📋 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)