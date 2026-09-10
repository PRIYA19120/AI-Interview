import pandas as pd
import re

from flask import Flask, render_template, request, jsonify, session
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==============================
# APP SETUP
# ==============================

app = Flask(__name__)
app.secret_key = "interview_secret_key"

# AI Model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# LOAD QUESTIONS DATASET
# ==============================

def load_questions():

    try:
        df = pd.read_csv("dataset/questions.csv")
    except Exception:

        try:
            df = pd.read_csv("dataset/question.csv")
        except Exception:
            df = pd.DataFrame()

    if not df.empty:
        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

    return df


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


# ==============================
# LOGIN PAGE
# ==============================

@app.route("/login")
def login():
    return render_template("login.html")


# ==============================
# INTERVIEW SETUP
# ==============================

@app.route("/interview-setup")
def interview_setup():
    return render_template("interview_setup.html")


# ==============================
# INTERVIEW PAGE
# ==============================

@app.route("/interview")
def interview():

    interview_type = request.args.get(
        "interview_type",
        "HR"
    ).strip()

    difficulty = request.args.get(
        "difficulty",
        "Intermediate"
    ).strip()

    try:
        limit = int(
            request.args.get(
                "num_questions",
                10
            )
        )
    except (ValueError, TypeError):
        limit = 10

    df = load_questions()

    questions_list = []

    if not df.empty:

        # Required columns
        if "type" in df.columns:
            df["type_clean"] = (
                df["type"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        if "difficulty" in df.columns:
            df["diff_clean"] = (
                df["difficulty"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

        interview_type_clean = interview_type.lower()
        difficulty_clean = difficulty.lower()

        # Exact type + difficulty
        if (
            "type_clean" in df.columns
            and "diff_clean" in df.columns
        ):

            filtered_df = df[
                (df["type_clean"] == interview_type_clean)
                &
                (df["diff_clean"] == difficulty_clean)
            ]

        else:
            filtered_df = df

        # Type fallback
        if filtered_df.empty and "type_clean" in df.columns:

            filtered_df = df[
                df["type_clean"] == interview_type_clean
            ]

        # Full dataset fallback
        if filtered_df.empty:
            filtered_df = df

        # Create question list
        for _, row in filtered_df.head(limit).iterrows():

            question_text = str(
                row.get("question", "")
            ).strip()

            if question_text:

                questions_list.append({

                    "question": question_text,

                    "type": str(
                        row.get(
                            "type",
                            interview_type
                        )
                    ).strip(),

                    "difficulty": str(
                        row.get(
                            "difficulty",
                            difficulty
                        )
                    ).strip(),

                    "reference_answer": str(
                        row.get(
                            "reference_answer",
                            ""
                        )
                    ).strip()
                })

    # Backup questions
    if not questions_list:

        questions_list = [

            {
                "question": "Tell me about yourself.",
                "type": "HR",
                "difficulty": "Beginner",
                "reference_answer":
                    "I am a motivated student with good communication "
                    "and problem solving skills."
            },

            {
                "question": "What are your strengths?",
                "type": "HR",
                "difficulty": "Beginner",
                "reference_answer":
                    "My strengths include communication, teamwork, "
                    "problem solving and willingness to learn."
            },

            {
                "question": "Why do you want this job?",
                "type": "HR",
                "difficulty": "Beginner",
                "reference_answer":
                    "I want this job because it provides an opportunity "
                    "to learn, contribute and grow professionally."
            }
        ]

    return render_template(
        "interview.html",
        questions=questions_list,
        total_questions=len(questions_list),
        interview_type=interview_type,
        difficulty=difficulty
    )


# ==============================
# ANSWER ANALYSIS
# ==============================

def analyze_answer(
    question,
    answer,
    reference_answer,
    interview_type="HR"
):

    answer = str(answer).strip()
    reference_answer = str(reference_answer).strip()

    # --------------------------------
    # Empty Answer
    # --------------------------------

    if not answer:

        return {

            "score": 0,
            "similarity": 0,
            "communication": 0,
            "technical": 0,
            "quality": 0,

            "feedback":
                "No answer was provided.",

            "strength":
                "No response",

            "improvement":
                "Please provide an answer."
        }


    # --------------------------------
    # No Reference Answer
    # --------------------------------

    if not reference_answer:

        return {

            "score": 0,
            "similarity": 0,
            "communication": 0,
            "technical": 0,
            "quality": 0,

            "feedback":
                "Reference answer is not available.",

            "strength":
                "Answer attempted",

            "improvement":
                "Add a reference answer for better evaluation."
        }


    # ==================================
    # 1. SEMANTIC RELEVANCE
    # ==================================

    embeddings = model.encode(
        [
            reference_answer,
            answer
        ]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    similarity_percent = round(
        max(0, min(1, similarity)) * 100
    )


    # ==================================
    # 2. COMMUNICATION SCORE
    # ==================================

    words = re.findall(
        r"\b\w+\b",
        answer
    )

    word_count = len(words)

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?]+",
            answer
        )
        if sentence.strip()
    ]

    sentence_count = len(sentences)


    # Length score
    if word_count >= 80:
        length_score = 100

    elif word_count >= 50:
        length_score = 90

    elif word_count >= 30:
        length_score = 80

    elif word_count >= 15:
        length_score = 65

    elif word_count >= 5:
        length_score = 45

    else:
        length_score = 20


    # Sentence structure
    if sentence_count >= 4:
        structure_score = 100

    elif sentence_count == 3:
        structure_score = 90

    elif sentence_count == 2:
        structure_score = 75

    elif sentence_count == 1:
        structure_score = 55

    else:
        structure_score = 20


    communication = round(
        (length_score * 0.60)
        +
        (structure_score * 0.40)
    )

    communication = max(
        0,
        min(100, communication)
    )


    # ==================================
    # 3. TECHNICAL UNDERSTANDING
    # ==================================

    if interview_type.lower() == "technical":

        technical = round(
            similarity_percent * 0.80
            +
            communication * 0.20
        )

    else:

        technical = round(
            similarity_percent * 0.60
            +
            communication * 0.40
        )

    technical = max(
        0,
        min(100, technical)
    )


    # ==================================
    # 4. ANSWER QUALITY
    # ==================================

    quality = round(

        similarity_percent * 0.45
        +
        communication * 0.30
        +
        technical * 0.25

    )

    quality = max(
        0,
        min(100, quality)
    )


    # ==================================
    # 5. OVERALL SCORE
    # ==================================

    score = round(

        similarity_percent * 0.50
        +
        communication * 0.20
        +
        technical * 0.15
        +
        quality * 0.15

    )

    score = max(
        0,
        min(100, score)
    )


    # ==================================
    # 6. FEEDBACK
    # ==================================

    if score >= 80:

        feedback = (
            "Excellent answer! Your response is highly relevant, "
            "clear and demonstrates strong understanding."
        )

        strength = (
            "Strong understanding and effective communication"
        )

        improvement = (
            "Keep practicing and add practical examples "
            "to make your answers even stronger."
        )

    elif score >= 60:

        feedback = (
            "Good answer. Your response is relevant, "
            "but some points could be explained in more detail."
        )

        strength = (
            "Good understanding of the topic"
        )

        improvement = (
            "Add more relevant details and practical examples."
        )

    elif score >= 40:

        feedback = (
            "Your answer is partially relevant, "
            "but it needs more explanation and clarity."
        )

        strength = (
            "You identified some relevant concepts"
        )

        improvement = (
            "Focus on the main concept and explain it more clearly."
        )

    else:

        feedback = (
            "Your answer has low relevance to the expected answer."
        )

        strength = (
            "You attempted the question"
        )

        improvement = (
            "Review the topic and focus on the main concepts."
        )


    return {

        "score": score,

        "similarity":
            similarity_percent,

        "communication":
            communication,

        "technical":
            technical,

        "quality":
            quality,

        "feedback":
            feedback,

        "strength":
            strength,

        "improvement":
            improvement
    }


# ==============================
# SUBMIT INTERVIEW
# ==============================

@app.route(
    "/submit-interview",
    methods=["POST"]
)
def submit_interview():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "error": "No data received."
        })


    questions = data.get(
        "questions",
        []
    )

    answers = data.get(
        "answers",
        {}
    )

    interview_type = data.get(
        "interview_type",
        "HR"
    )

    difficulty = data.get(
        "difficulty",
        "Intermediate"
    )


    df = load_questions()

    results = []


    # ==================================
    # ANALYZE EACH ANSWER
    # ==================================

    for index, question_data in enumerate(questions):

        question_text = str(
            question_data.get(
                "question",
                ""
            )
        ).strip()


        user_answer = str(
            answers.get(
                str(index),
                ""
            )
        ).strip()


        reference_answer = ""


        # Find reference answer
        if (
            not df.empty
            and "reference_answer" in df.columns
            and "question" in df.columns
        ):

            matching_rows = df[
                df["question"]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                question_text.lower()
            ]


            if not matching_rows.empty:

                reference_answer = str(
                    matching_rows.iloc[0][
                        "reference_answer"
                    ]
                ).strip()


        # AI Analysis
        analysis = analyze_answer(

            question_text,

            user_answer,

            reference_answer,

            interview_type
        )


        results.append({

            "question":
                question_text,

            "answer":
                user_answer,

            "reference_answer":
                reference_answer,

            "score":
                analysis["score"],

            "similarity":
                analysis["similarity"],

            "communication":
                analysis["communication"],

            "technical":
                analysis["technical"],

            "quality":
                analysis["quality"],

            "feedback":
                analysis["feedback"],

            "strength":
                analysis["strength"],

            "improvement":
                analysis["improvement"]
        })


    # ==================================
    # OVERALL SCORE
    # ==================================

    if results:

        overall_score = round(

            sum(
                item["score"]
                for item in results
            )
            /
            len(results)

        )

    else:

        overall_score = 0


    # ==================================
    # RESPONSE COUNTS
    # ==================================

    excellent_responses = sum(

        1
        for item in results
        if item["score"] >= 80

    )


    good_responses = sum(

        1
        for item in results
        if 60 <= item["score"] < 80

    )


    need_improvement = sum(

        1
        for item in results
        if item["score"] < 60

    )


    # ==================================
    # SESSION SAVE
    # ==================================

    session["interview_results"] = results

    session["overall_score"] = overall_score

    # Save interview score history
    score_history = session.get("score_history", [])

    score_history.append(overall_score)

    session["score_history"] = score_history

    # Best score
    best_score = max(score_history)

    # Score improvement
    if len(score_history) > 1:
        improvement = overall_score - score_history[-2]
    else:
        improvement = 0

    session["best_score"] = best_score
    session["score_improvement"] = improvement

    session["excellent_responses"] = excellent_responses
    session["good_responses"] = good_responses
    session["need_improvement"] = need_improvement
    session["total_questions"] = len(results)
    session["interview_type"] = interview_type
    session["difficulty"] = difficulty

    # ==================================
    # DEBUG
    # ==================================

    print("\n==============================")
    print("INTERVIEW ANALYSIS")
    print("==============================")
    print("Interview Type:", interview_type)
    print("Difficulty:", difficulty)
    print("Total Questions:", len(results))
    print("Overall Score:", overall_score)
    print("Excellent:", excellent_responses)
    print("Good:", good_responses)
    print("Need Improvement:", need_improvement)
    print("Results:", results)
    print("==============================\n")

    return jsonify({
        "success": True,
        "score": overall_score,
        "excellent_responses": excellent_responses,
        "good_responses": good_responses,
        "need_improvement": need_improvement,
        "total_questions": len(results)
    })
# SKILLS ANALYSIS PAGE
# ==============================

@app.route("/skills")
def skills():

    results = session.get(
        "interview_results",
        []
    )

    total_questions = len(results)


    # ==================================
    # NO INTERVIEW RESULTS
    # ==================================

    if total_questions == 0:

        return render_template(
            "skills.html",

            overall_score=0,

            excellent_responses=0,

            good_responses=0,

            strong_responses=0,

            need_improvement=0,

            total_questions=0,

            relevance=0,

            communication=0,

            technical=0,

            quality=0,

            strengths=[
                {
                    "title": "Start an Interview",
                    "text": "Complete an interview to see your skill analysis."
                }
            ],

            improvements=[],

            interview_type=session.get(
                "interview_type",
                "HR"
            ),

            difficulty=session.get(
                "difficulty",
                "Intermediate"
            )
        )


    # ==================================
    # GET INDIVIDUAL SCORES
    # ==================================

    scores = [
        float(
            result.get(
                "score",
                0
            )
        )
        for result in results
    ]


    # ==================================
    # STRONGEST & WEAKEST ANSWERS
    # ==================================

    sorted_results = sorted(
        results,
        key=lambda x: float(
            x.get(
                "score",
                0
            )
        ),
        reverse=True
    )

    # Top 3 strongest answers
    strongest_answers = sorted_results[:3]

    # Top 3 answers needing improvement
    weakest_answers = sorted_results[-3:]

    # Lowest score first
    weakest_answers = sorted(
        weakest_answers,
        key=lambda x: float(
            x.get(
                "score",
                0
            )
        )
    )


    # ==================================
    # OVERALL SCORE
    # ==================================

    overall_score = round(
        sum(scores) / len(scores)
    )


    # ==================================
    # RESPONSE CATEGORIES
    # ==================================

    # 80 - 100 = Excellent
    excellent_responses = sum(
        1
        for score in scores
        if score >= 80
    )


    # 60 - 79 = Good
    good_responses = sum(
        1
        for score in scores
        if 60 <= score < 80
    )


    # Below 60 = Need Improvement
    need_improvement = sum(
        1
        for score in scores
        if score < 60
    )


    # Strong Responses
    # Excellent + Good
    strong_responses = (
        excellent_responses
        +
        good_responses
    )


    # ==================================
    # ANSWER RELEVANCE
    # ==================================

    relevance = round(
        sum(
            float(
                result.get(
                    "similarity",
                    0
                )
            )
            for result in results
        )
        / len(results)
    )


    # ==================================
    # COMMUNICATION
    # ==================================

    communication = round(
        sum(
            float(
                result.get(
                    "communication",
                    0
                )
            )
            for result in results
        )
        / len(results)
    )


    # ==================================
    # TECHNICAL UNDERSTANDING
    # ==================================

    technical = round(
        sum(
            float(
                result.get(
                    "technical",
                    0
                )
            )
            for result in results
        )
        / len(results)
    )


    # ==================================
    # ANSWER QUALITY
    # ==================================

    quality = round(
        sum(
            float(
                result.get(
                    "quality",
                    0
                )
            )
            for result in results
        )
        / len(results)
    )


    # ==================================
    # STRENGTHS
    # ==================================

    strengths = []


    if relevance >= 60:

        strengths.append(
            {
                "title": "Answer Relevance",
                "text": (
                    "Your answers are relevant "
                    "to the questions asked."
                )
            }
        )


    if communication >= 60:

        strengths.append(
            {
                "title": "Communication",
                "text": (
                    "Your answers are clear "
                    "and well structured."
                )
            }
        )


    if technical >= 60:

        strengths.append(
            {
                "title": "Technical Understanding",
                "text": (
                    "You demonstrate a good "
                    "understanding of the concepts."
                )
            }
        )


    if quality >= 60:

        strengths.append(
            {
                "title": "Answer Quality",
                "text": (
                    "Your answers contain useful "
                    "and relevant information."
                )
            }
        )


    # If no strong area
    if not strengths:

        strengths.append(
            {
                "title": "Keep Practicing",
                "text": (
                    "Continue practicing interview "
                    "questions to improve your responses."
                )
            }
        )


    # ==================================
    # AREAS TO IMPROVE
    # ==================================

    improvements = []


    if relevance < 60:

        improvements.append(
            {
                "title": "Answer Relevance",
                "text": (
                    "Focus on directly answering "
                    "the main concept asked in the question."
                )
            }
        )


    if communication < 60:

        improvements.append(
            {
                "title": "Communication",
                "text": (
                    "Try to give clearer and more "
                    "structured answers."
                )
            }
        )


    if technical < 60:

        improvements.append(
            {
                "title": "Technical Understanding",
                "text": (
                    "Review important concepts and "
                    "include relevant technical details."
                )
            }
        )


    if quality < 60:

        improvements.append(
            {
                "title": "Answer Detail",
                "text": (
                    "Try to explain your answers with "
                    "more relevant details and examples."
                )
            }
        )


    # ==================================
    # RENDER SKILLS PAGE
    # ==================================

    return render_template(

        "skills.html",

        strongest_answers=strongest_answers,

        weakest_answers=weakest_answers,

        overall_score=overall_score,

        excellent_responses=excellent_responses,

        good_responses=good_responses,

        strong_responses=strong_responses,

        need_improvement=need_improvement,

        total_questions=total_questions,

        relevance=relevance,

        communication=communication,

        technical=technical,

        quality=quality,

        strengths=strengths,

        improvements=improvements,

        interview_type=session.get(
            "interview_type",
            "HR"
        ),

        difficulty=session.get(
            "difficulty",
            "Intermediate"
        )
    )


# ==============================
# RESULT PAGE
# ==============================

@app.route("/result")
def result():

    results = session.get(
        "interview_results",
        []
    )


    overall_score = session.get(
        "overall_score",
        0
    )


    interview_type = session.get(
        "interview_type",
        "HR"
    )


    difficulty = session.get(
        "difficulty",
        "Intermediate"
    )


    return render_template(

        "result.html",

        results=results,

        overall_score=overall_score,

        interview_type=interview_type,

        difficulty=difficulty
    )


# ==============================
# PRACTICE PAGE
# ==============================

@app.route("/practice")
def practice():

    results = session.get("interview_results", [])
    overall_score = session.get("overall_score", 0)
    total_questions = session.get("total_questions", 0)

    excellent_responses = session.get(
        "excellent_responses", 0
    )

    good_responses = session.get(
        "good_responses", 0
    )

    need_improvement = session.get(
        "need_improvement", 0
    )

    interview_type = session.get(
        "interview_type", "HR"
    )

    difficulty = session.get(
        "difficulty", "Intermediate"
    )

    score_history = session.get("score_history", [])
    best_score = session.get("best_score", 0)
    score_improvement = session.get("score_improvement", 0)

    return render_template(
        "practice.html",
        results=results,
        overall_score=overall_score,
        total_questions=total_questions,
        excellent_responses=excellent_responses,
        good_responses=good_responses,
        need_improvement=need_improvement,
        interview_type=interview_type,
        difficulty=difficulty,
        score_history=score_history,
        best_score=best_score,
        score_improvement=score_improvement
    )


# ==============================
# HOW IT WORKS
# ==============================

@app.route("/how-it-works")
def how_it_works():

    return render_template(
        "how_it_work.html"
    )


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )
