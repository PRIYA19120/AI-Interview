import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)
app.secret_key = "interview_secret_key"

# 1. Home Page
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

# 2. Login Page
@app.route('/login')
def login():
    return render_template('login.html')

# 3. Interview Setup Page
@app.route('/interview-setup')
def interview_setup():
    return render_template('interview_setup.html')

# 4. Main Interview Page
@app.route('/interview')
def interview():
    # Setup page se aane wale values
    itype = request.args.get('interview_type', 'HR').strip().lower()
    diff = request.args.get('difficulty', 'Intermediate').strip().lower()
    
    try:
        limit = int(request.args.get('num_questions', 10))
    except (ValueError, TypeError):
        limit = 10

    # Direct CSV read (dataset folder se)
    try:
        df = pd.read_csv("dataset/questions.csv")
    except Exception:
        try:
            df = pd.read_csv("dataset/question.csv")
        except Exception:
            df = pd.DataFrame()

    questions_list = []

    if not df.empty:
        # Columns ke naam lower case karna
        df.columns = [c.strip().lower() for c in df.columns]

        # String formatting for matching
        df['type_clean'] = df['type'].astype(str).str.strip().str.lower()
        df['diff_clean'] = df['difficulty'].astype(str).str.strip().str.lower()

        # Step 1: Type aur Difficulty dono filter karna
        filtered_df = df[(df['type_clean'] == itype) & (df['diff_clean'] == diff)]

        # Step 2: Agar exact match na mile toh sirf Type filter karo
        if filtered_df.empty:
            filtered_df = df[df['type_clean'] == itype]

        # Step 3: Agar abhi bhi empty ho toh pura dataframe use karo
        if filtered_df.empty:
            filtered_df = df

        # Question list build karna
        for _, row in filtered_df.head(limit).iterrows():
            q_text = str(row.get('question', '')).strip()
            q_type = str(row.get('type', itype.capitalize())).strip()
            q_diff = str(row.get('difficulty', diff.capitalize())).strip()
            
            if q_text:
                questions_list.append({
                    'question': q_text,
                    'type': q_type,
                    'difficulty': q_diff,
                    'reference_answer': str(
                        row.get('reference_answer', '')
                    ).strip()
                })

    # Backup questions agar CSV khali ho
    if not questions_list:
        questions_list = [
            {'question': 'Tell me about yourself.', 'type': 'HR', 'difficulty': 'Beginner'},
            {'question': 'What are your strengths?', 'type': 'HR', 'difficulty': 'Beginner'},
            {'question': 'Why do you want this job?', 'type': 'HR', 'difficulty': 'Beginner'}
        ]

    return render_template(
        'interview.html',
        questions=questions_list,
        total_questions=len(questions_list),
        interview_type=request.args.get('interview_type', 'HR'),
        difficulty=request.args.get('difficulty', 'Intermediate')
    )
def analyze_answer(question, answer, reference_answer):

    answer = str(answer).strip()
    reference_answer = str(reference_answer).strip()

    if not answer:
        return {
            "score": 0,
            "similarity": 0,
            "feedback": "No answer was provided.",
            "strength": "No response",
            "improvement": "Please provide an answer."
        }

    if not reference_answer:
        return {
            "score": 0,
            "similarity": 0,
            "feedback": "Reference answer is not available.",
            "strength": "Cannot evaluate",
            "improvement": "Add a reference answer for this question."
        }

    # Convert answers into semantic embeddings
    embeddings = model.encode([
        reference_answer,
        answer
    ])

    # Calculate semantic similarity
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    # Convert similarity into percentage
    similarity_percent = round(max(0, similarity) * 100)

    # Score based on semantic similarity
    score = similarity_percent

    if score >= 80:

        feedback = (
            "Excellent answer! Your answer is highly relevant "
            "and shows a strong understanding of the topic."
        )

        strength = (
            "Strong understanding and high relevance"
        )

        improvement = (
            "Try adding a practical example to make your answer "
            "even stronger."
        )

    elif score >= 60:

        feedback = (
            "Good answer. Your response is relevant, "
            "but some important points could be explained better."
        )

        strength = (
            "Good understanding of the topic"
        )

        improvement = (
            "Add more relevant details or a practical example."
        )

    elif score >= 40:

        feedback = (
            "Your answer is partially relevant, "
            "but it needs more explanation."
        )

        strength = (
            "Some relevant concepts were identified"
        )

        improvement = (
            "Explain the main idea more clearly and "
            "include important concepts."
        )

    else:

        feedback = (
            "Your answer has low relevance to the expected answer."
        )

        strength = (
            "You attempted the question"
        )

        improvement = (
            "Review the topic and focus on the main concepts "
            "asked in the question."
        )

    return {
        "score": score,
        "similarity": similarity_percent,
        "feedback": feedback,
        "strength": strength,
        "improvement": improvement
    }
@app.route('/submit-interview', methods=['POST'])
def submit_interview():

    data = request.get_json()

    questions = data.get('questions', [])
    answers = data.get('answers', {})
    interview_type = data.get('interview_type', '')
    difficulty = data.get('difficulty', '')

    # CSV load karo
    try:
        df = pd.read_csv("dataset/questions.csv")
    except Exception:
        try:
            df = pd.read_csv("dataset/question.csv")
        except Exception:
            df = pd.DataFrame()

    results = []

    # CSV columns clean
    if not df.empty:
        df.columns = [c.strip().lower() for c in df.columns]

    # Har question ke answer ko analyze karo
    for index, question_data in enumerate(questions):

        question_text = str(
            question_data.get('question', '')
        ).strip()

        # User answer
        user_answer = str(
            answers.get(str(index), '')
        ).strip()

        # Reference answer
        reference_answer = ""

        if not df.empty and 'reference_answer' in df.columns:

            matching_rows = df[
                df['question']
                .astype(str)
                .str.strip()
                .str.lower()
                == question_text.lower()
            ]

            if not matching_rows.empty:
                reference_answer = str(
                    matching_rows.iloc[0]['reference_answer']
                ).strip()

        # ML analysis
        analysis = analyze_answer(
            question_text,
            user_answer,
            reference_answer
        )

        results.append({
            "question": question_text,
            "answer": user_answer,
            "reference_answer": reference_answer,
            "score": analysis["score"],
            "similarity": analysis["similarity"],
            "feedback": analysis["feedback"],
            "strength": analysis["strength"],
            "improvement": analysis["improvement"]
        })

    # Overall score
    if results:
        overall_score = round(
            sum(item["score"] for item in results)
            / len(results)
        )
    else:
        overall_score = 0

    # Session mein complete analysis save karo
    session['interview_results'] = results
    session['overall_score'] = overall_score
    session['interview_type'] = interview_type
    session['difficulty'] = difficulty

    # Debug ke liye terminal mein print
    print("Interview Type:", interview_type)
    print("Difficulty:", difficulty)
    print("Overall Score:", overall_score)
    print("Interview Results:", results)

    return jsonify({
        "success": True,
        "score": overall_score
    })
# 5. Skills Analysis Page
@app.route('/skills')
def skills():

    results = session.get('interview_results', [])
    overall_score = session.get('overall_score', 0)

    interview_type = session.get(
        'interview_type',
        'HR'
    )

    difficulty = session.get(
        'difficulty',
        'Intermediate'
    )

    total_questions = len(results)

    # Response categories
    excellent_responses = sum(
        1 for item in results
        if item.get('score', 0) >= 80
    )

    good_responses = sum(
        1 for item in results
        if 60 <= item.get('score', 0) < 80
    )

    strong_responses = sum(
        1 for item in results
        if item.get('score', 0) >= 60
    )

    need_improvement = sum(
        1 for item in results
        if item.get('score', 0) < 60
    )

    # Average answer quality
    if total_questions > 0:
        answer_quality = round(
            sum(
                item.get('score', 0)
                for item in results
            ) / total_questions
        )
    else:
        answer_quality = 0

    # Communication score
    # Based on answer length
    communication_scores = []

    for item in results:

        answer = str(
            item.get('answer', '')
        ).strip()

        word_count = len(answer.split())

        if word_count >= 30:
            communication_scores.append(90)

        elif word_count >= 15:
            communication_scores.append(75)

        elif word_count >= 5:
            communication_scores.append(60)

        else:
            communication_scores.append(35)

    if communication_scores:
        communication_score = round(
            sum(communication_scores)
            / len(communication_scores)
        )
    else:
        communication_score = 0

    # Technical understanding
    # Uses ML similarity score
    technical_scores = [
        item.get('similarity', 0)
        for item in results
    ]

    if technical_scores:
        technical_score = round(
            sum(technical_scores)
            / len(technical_scores)
        )
    else:
        technical_score = 0

    return render_template(
        'skills.html',

        overall_score=overall_score,

        strong_responses=strong_responses,

        need_improvement=need_improvement,

        total_questions=total_questions,

        excellent_responses=excellent_responses,

        good_responses=good_responses,

        answer_quality=answer_quality,

        communication_score=communication_score,

        technical_score=technical_score,

        interview_type=interview_type,

        difficulty=difficulty
    )

# 6. Result Page
@app.route('/result')
def result():

    results = session.get('interview_results', [])
    overall_score = session.get('overall_score', 0)

    interview_type = session.get(
        'interview_type',
        'HR'
    )

    difficulty = session.get(
        'difficulty',
        'Intermediate'
    )

    return render_template(
        'result.html',
        results=results,
        overall_score=overall_score,
        interview_type=interview_type,
        difficulty=difficulty
    )
# 7. Practice Page
@app.route('/practice')
def practice():
    return render_template('practice.html')

# 8. How It Works Page
@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_work.html')

if __name__ == '__main__':
    app.run(debug=True)