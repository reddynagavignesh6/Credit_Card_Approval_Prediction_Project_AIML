# ============================================================
# CREDIT CARD APPROVAL PREDICTION USING MACHINE LEARNING
# ============================================================

from flask import Flask, render_template, request
from datetime import datetime

import pickle
import sqlite3
import numpy as np
import pandas as pd

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# ============================================================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # APPLICANT NAME
        # ----------------------------------------------------

        applicant_name = request.form["APPLICANT_NAME"]

        # ----------------------------------------------------
        # COLLECT INPUT FEATURES
        # ----------------------------------------------------

        features = [

            float(request.form["CODE_GENDER"]),
            float(request.form["FLAG_OWN_CAR"]),
            float(request.form["FLAG_OWN_REALTY"]),
            float(request.form["CNT_CHILDREN"]),
            float(request.form["AMT_INCOME_TOTAL"]),
            float(request.form["NAME_INCOME_TYPE"]),
            float(request.form["NAME_EDUCATION_TYPE"]),
            float(request.form["NAME_FAMILY_STATUS"]),
            float(request.form["NAME_HOUSING_TYPE"]),
            float(request.form["DAYS_BIRTH"]),
            float(request.form["DAYS_EMPLOYED"]),
            float(request.form["FLAG_MOBIL"]),
            float(request.form["FLAG_WORK_PHONE"]),
            float(request.form["FLAG_PHONE"]),
            float(request.form["FLAG_EMAIL"]),
            float(request.form["OCCUPATION_TYPE"]),
            float(request.form["CNT_FAM_MEMBERS"])

        ]

        # ----------------------------------------------------
        # CONVERT INPUT TO DATAFRAME
        # (Removes sklearn feature-name warning)
        # ----------------------------------------------------

        columns = [

            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "FLAG_MOBIL",
            "FLAG_WORK_PHONE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "OCCUPATION_TYPE",
            "CNT_FAM_MEMBERS"

        ]

        features = pd.DataFrame([features], columns=columns)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(features)[0]

        probability = model.predict_proba(features)

        confidence = round(np.max(probability) * 100, 2)

        # ----------------------------------------------------
        # PREDICTION RESULT
        # ----------------------------------------------------

        if prediction == 0:

            result = "Good Customer"

            approval = "Credit Card Approved"

            icon = "✅"

        else:

            result = "Bad Customer"

            approval = "Credit Card Rejected"

            icon = "❌"

        # ----------------------------------------------------
        # DATE & TIME
        # ----------------------------------------------------

        current_date = datetime.now().strftime("%d-%m-%Y")

        current_time = datetime.now().strftime("%I:%M %p")
                # ----------------------------------------------------
        # SAVE PREDICTION TO SQLITE DATABASE
        # ----------------------------------------------------

        connection = sqlite3.connect("credit_card.db")

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO predictions
            (
                applicant_name,
                prediction,
                confidence,
                prediction_date,
                prediction_time
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                applicant_name,
                result,
                confidence,
                current_date,
                current_time
            )
        )

        connection.commit()

        connection.close()

        # ----------------------------------------------------
        # DISPLAY RESULT PAGE
        # ----------------------------------------------------

        return render_template(

            "result.html",

            applicant_name=applicant_name,

            prediction=result,

            approval=approval,

            icon=icon,

            confidence=confidence,

            prediction_date=current_date,

            prediction_time=current_time

        )

    except Exception as e:

        return render_template(

            "result.html",

            prediction=f"Error : {str(e)}"

        )


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.route("/history")
def history():

    connection = sqlite3.connect("credit_card.db")

    cursor = connection.cursor()

    cursor.execute("""

    SELECT
        id,
        applicant_name,
        prediction,
        confidence,
        prediction_date,
        prediction_time

    FROM predictions

    ORDER BY id DESC

    """)

    history_data = cursor.fetchall()

    connection.close()

    return render_template(

        "history.html",

        history=history_data

    )
# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    connection = sqlite3.connect("credit_card.db")

    cursor = connection.cursor()

    # ----------------------------------------------------
    # TOTAL PREDICTIONS
    # ----------------------------------------------------

    cursor.execute("SELECT COUNT(*) FROM predictions")

    total = cursor.fetchone()[0]

    # ----------------------------------------------------
    # APPROVED CUSTOMERS
    # ----------------------------------------------------

    cursor.execute(

        "SELECT COUNT(*) FROM predictions WHERE prediction='Good Customer'"

    )

    approved = cursor.fetchone()[0]

    # ----------------------------------------------------
    # REJECTED CUSTOMERS
    # ----------------------------------------------------

    cursor.execute(

        "SELECT COUNT(*) FROM predictions WHERE prediction='Bad Customer'"

    )

    rejected = cursor.fetchone()[0]

    # ----------------------------------------------------
    # AVERAGE CONFIDENCE
    # ----------------------------------------------------

    cursor.execute(

        "SELECT AVG(confidence) FROM predictions"

    )

    average_confidence = cursor.fetchone()[0]

    if average_confidence is None:

        average_confidence = 0

    average_confidence = round(average_confidence, 2)

    # ----------------------------------------------------
    # APPROVAL RATE
    # ----------------------------------------------------

    approval_rate = 0

    if total > 0:

        approval_rate = round((approved / total) * 100, 2)

    connection.close()

    # ----------------------------------------------------
    # DISPLAY DASHBOARD
    # ----------------------------------------------------

    return render_template(

        "dashboard.html",

        total=total,

        approved=approved,

        rejected=rejected,

        approval_rate=approval_rate,

        average_confidence=average_confidence

    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )