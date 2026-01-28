from flask import Flask, render_template, request, session, jsonify, flash
from utils import preprocess_and_save, save_analysis_history
import pandas as pd
from groq import Groq
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
import time
from dateutil import relativedelta
import io

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

@app.template_filter('datetime')
def format_datetime(value, format='%Y-%m-%d %H:%M:%S'):
    """Format a datetime object or ISO string to readable format"""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except:
            return value
    if isinstance(value, datetime):
        return value.strftime(format)
    return value

@app.template_filter('time_ago')
def time_ago_filter(value):
    """Convert datetime to human-readable time ago format"""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except:
            return value
    
    if not isinstance(value, datetime):
        return value
    
    now = datetime.now(timezone.utc) if value.tzinfo else datetime.now()
    diff = relativedelta.relativedelta(now, value)
    
    if diff.years > 0:
        return f"{diff.years} year{'s' if diff.years > 1 else ''} ago"
    elif diff.months > 0:
        return f"{diff.months} month{'s' if diff.months > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.hours > 0:
        return f"{diff.hours} hour{'s' if diff.hours > 1 else ''} ago"
    elif diff.minutes > 0:
        return f"{diff.minutes} minute{'s' if diff.minutes > 1 else ''} ago"
    else:
        return "Just now"

@app.template_filter('first')
def first_filter(value):
    """Get first element of iterable"""
    if hasattr(value, '__iter__') and not isinstance(value, str):
        try:
            return next(iter(value))
        except:
            return value
    return value

@app.route("/")
def index():
    total_analyzed = 1000  # Default base
    try:
        if os.path.exists('analysis_history.json'):
            with open('analysis_history.json', 'r') as f:
                history = json.load(f)
                total_analyzed += len(history)
    except:
        pass
    return render_template("index.html", total_analyzed=total_analyzed)

@app.route("/dashboard")
def dashboard():
    file_path = session.get('current_df_path')
    stats = None
    
    # Performance metrics from history (always calculate if exists)
    performance = {"success_rate": 0, "avg_time": 0.0, "total": 0}
    try:
        if os.path.exists('analysis_history.json'):
            with open('analysis_history.json', 'r') as f:
                history = json.load(f)
                if history:
                    performance["total"] = len(history)
                    successes = [h for h in history if h.get('success', True)]
                    performance["success_rate"] = round((len(successes) / len(history)) * 100, 1)
                    times = [h.get('response_time', 0) for h in history if h.get('response_time')]
                    if times:
                        performance["avg_time"] = round(sum(times) / len(times), 2)
    except:
        pass

    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            null_total = df.isnull().sum().sum()
            total_cells = df.size
            completeness = round(((total_cells - null_total) / total_cells) * 100, 2) if total_cells > 0 else 0
            
            stats = {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_cols": len(df.select_dtypes(include=['number']).columns),
                "categorical_cols": len(df.select_dtypes(include=['object']).columns),
                "completeness": completeness,
                "memory": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                "null_counts": df.isnull().sum().to_dict(),
                "performance": performance
            }
        except:
            pass
    
    # If no data loaded but history exists, still pass performance
    if not stats:
        stats = {"performance": performance}
        
    return render_template("dashboard.html", stats=stats)

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        file = request.files.get("file")
        query = request.form.get("query")
        
        if not file:
            file_path = session.get('current_df_path')
            if not file_path or not os.path.exists(file_path):
                return jsonify({"error": "Please upload a file"})
            df = pd.read_csv(file_path)
        else:
            df, cols, df_html, file_path, err = preprocess_and_save(file)
            if err:
                return jsonify({"error": err})
            session['current_df_path'] = file_path
            session['columns'] = cols
        
        if query:
            try:
                # Prepare context for the LLM
                buffer = io.StringIO()
                df.info(buf=buffer)
                df_info = buffer.getvalue()
                
                prompt = f"""
You are an expert Python data analyst.
You have a pandas DataFrame named `df` loaded with the following structure:

Columns: {list(df.columns)}
Data Types and Info:
{df_info}

Sample Data (first 3 rows):
{df.head(3).to_string()}

User Question: {query}

**Task**: Write Python code using pandas to answer the user's question.
**Rules**:
1. Assume `df` is already loaded.
2. Use `result` as the final output variable. if the answer is a single value, assign it to `result`. if it's a plot, you can't display it, so just return the data for the plot in `result`.
3. Handle potential NaN values gracefully.
4. IMPORT NOTHING. `pandas` is available as `pd`.
5. RETURN ONLY THE CODE. No markdown backticks, no `python` keyword, no explanations. Just valid Python code.
"""
                client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                start_time = time.time()
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1
                )

                code_generated = chat_completion.choices[0].message.content.strip("`python").strip("`").strip()
                
                local_vars = {"df": df, "pd": pd}
                exec(code_generated, {}, local_vars)
                
                result = local_vars.get("result")
                end_time = time.time()
                response_time = round(end_time - start_time, 2)
                
                # Save to history
                history_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "code": code_generated,
                    "result": result.to_dict() if isinstance(result, pd.DataFrame) else str(result),
                    "response_time": response_time,
                    "success": True
                }
                save_analysis_history(history_entry)
                
                if isinstance(result, pd.DataFrame):
                    result_html = result.to_html(classes="table table-striped", index=False)
                else:
                    result_html = f"<div class='result-value'>{result}</div>"
                
                return jsonify({
                    "success": True,
                    "code": code_generated,
                    "result": result_html,
                    "preview": df.head(10).to_html(classes="table table-striped", index=False)
                })
                
            except Exception as e:
                error_msg = str(e)
                if "401" in error_msg or "api_key" in error_msg.lower():
                    error_msg = "Invalid Groq API Key. Please check your .env file or Settings."
                elif "limit" in error_msg.lower():
                    error_msg = "API Rate limit exceeded. Please try again in a moment."
                return jsonify({"error": f"Error processing query: {error_msg}"})
    
    return render_template("analyze.html")

@app.route("/history")
def history():
    try:
        with open('analysis_history.json', 'r') as f:
            history = json.load(f)
    except:
        history = []
    return render_template("history.html", history=history)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        flash('Settings updated successfully!', 'success')
    return render_template("settings.html", groq_api_key=os.getenv('GROQ_API_KEY', ''))

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/api/data_stats")
def data_stats():
    file_path = session.get('current_df_path')
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
        null_total = df.isnull().sum().sum()
        total_cells = df.size
        completeness = round(((total_cells - null_total) / total_cells) * 100, 2) if total_cells > 0 else 0
        
        # Performance metrics from history
        performance = {"success_rate": 100, "avg_time": 0.0, "total": 0}
        try:
            if os.path.exists('analysis_history.json'):
                with open('analysis_history.json', 'r') as f:
                    history = json.load(f)
                    if history:
                        performance["total"] = len(history)
                        successes = [h for h in history if h.get('success', True)]
                        performance["success_rate"] = round((len(successes) / len(history)) * 100, 1)
                        times = [h.get('response_time', 0) for h in history if h.get('response_time')]
                        if times:
                            performance["avg_time"] = round(sum(times) / len(times), 2)
        except:
            pass

        stats = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_cols": len(df.select_dtypes(include=['number']).columns),
            "categorical_cols": len(df.select_dtypes(include=['object']).columns),
            "completeness": completeness,
            "memory": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            "null_counts": df.isnull().sum().to_dict(),
            "cols": df.columns.tolist(),
            "performance": performance
        }
        return jsonify(stats)
    return jsonify({"error": "No data loaded"})

@app.route("/api/clear_data", methods=["POST"])
def clear_data():
    file_path = session.get('current_df_path')
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            pass
    session.pop('current_df_path', None)
    session.pop('columns', None)
    return jsonify({"success": True})

@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    try:
        with open('analysis_history.json', 'w') as f:
            json.dump([], f)
        return jsonify({"success": True})
    except:
        return jsonify({"error": "Could not clear history"})

@app.route("/api/delete_history/<int:index>", methods=["DELETE"])
def delete_history(index):
    try:
        if os.path.exists('analysis_history.json'):
            with open('analysis_history.json', 'r') as f:
                history = json.load(f)
            
            if 0 <= index < len(history):
                history.pop(index)
                
            with open('analysis_history.json', 'w') as f:
                json.dump(history, f)
            return jsonify({"success": True})
    except:
        pass
    return jsonify({"error": "Could not delete entry"})

@app.route("/api/history_recent")
def history_recent():
    try:
        if os.path.exists('analysis_history.json'):
            with open('analysis_history.json', 'r') as f:
                history = json.load(f)
            return jsonify(list(reversed(history[-5:])))
    except:
        pass
    return jsonify([])


if __name__ == "__main__":
    app.run(debug=True)