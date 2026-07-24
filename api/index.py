import os
import json
import traceback
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import httpx
from groq import Groq

# Force load environment variables from the project root workspace directory
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

app = Flask(__name__, template_folder='../templates')

# Pre-flight verification check to help diagnose local errors in the terminal console
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    print("\n[CRITICAL WARNING] 'GROQ_API_KEY' environment variable was not found!")
    print("Ensure your .env file is positioned in the project root directory and contains a valid key.\n")
else:
    print(f"\n[SYSTEM INITIALIZATION] Groq API client verified. Key Prefix: {api_key[:6]}...\n")

# Initialize the uncompromised HTTP client wrapper configuration
groq_client = Groq(
    api_key=api_key,
    http_client=httpx.Client()
)

@app.route('/')
def index():
    """Renders the single-page interactive glassmorphic dashboard."""
    return render_template(
        'index.html',
        supabase_url=os.environ.get('SUPABASE_URL', ''),
        supabase_anon_key=os.environ.get('SUPABASE_ANON_KEY', '')
    )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Core health check route to verify engine operation and connectivity."""
    return jsonify({
        "status": "healthy",
        "message": "PathFinder AI API layer is fully responsive.",
        "environment": "vercel-serverless"
    })

@app.route('/api/navigate', methods=['POST'])
def generate_strategy_matrix():
    """
    Accepts user DNA criteria profiles and processes them through Groq's high-speed
    inference layer to output an optimized structural US career navigation matrix.
    """
    try:
        data = request.get_json() or {}
        
        academic_baseline = data.get('academicBaseline', '')
        work_style = data.get('workStyle', '')
        selected_skills = data.get('selectedSkills', [])
        geolocation = data.get('geolocation', '')
        custom_path = data.get('customPath', '')

        if not academic_baseline or not work_style or not selected_skills or not geolocation:
            return jsonify({"error": "Missing required strategic onboarding inputs."}), 400

        skills_csv = ", ".join(selected_skills)
        
        # Rigorous US economic matrix tailoring configuration rules
        system_instruction = (
            "You are an expert AI/ML Solutions Architect and United States macro-economic strategy engine specializing in career navigation intelligence.\n"
            "Your objective is to ingest a US student or professional's career DNA parameters and synthesize 3 distinct, tailored alternative US career paths.\n"
            "All market metrics, growth paths, and salary figures must reflect strict US cost-of-living metrics, US baseline benchmarks, and US company expectations.\n"
            "You MUST respond exclusively with a single valid JSON object. Do not include markdown formatting code blocks, no trailing text, no preambles.\n\n"
            "The JSON structure must exactly replicate the following template:\n"
            "{\n"
            '  "careers": [\n'
            "    {\n"
            '      "title": "Exact US Job Title Name",\n'
            '      "alignment_type": "Choose exactly either \'Optimal Alignment\', \'High Growth\', or \'Balanced Track\'",\n'
            '      "description": "Deep analytical breakdown of this alternative trajectory within the US economic marketplace.",\n'
            '      "future_proof_index": 92,\n' 
            '      "automation_risk": "Choose exactly either \'Low\', \'Moderate\', or \'High\'",\n'
            '      "market_trend_title": "US Sector Alert Title",\n'
            '      "market_trend_description": "Detailed digest of macro shifts, US Department of Labor occupational updates, and regional US hiring data.",\n'
            '      "certifications": ["US Cert 1 (e.g., AWS, CompTIA Security+, PMP)", "US Cert 2"],\n'
            '      "timeline": {\n'
            '        "year_1": {"title": "Junior Title", "salary": "$XX,XXX - $XX,XXX", "percentage": 45},\n' 
            '        "year_5": {"title": "Senior Title", "salary": "$XXX,XXX - $XXX,XXX", "percentage": 70},\n'
            '        "year_10": {"title": "Principal/Director Title", "salary": "$XXX,XXX - $XXX,XXX", "percentage": 95}\n'
            '      },\n'
            '      "day_in_life": [\n'
            '        {"time": "09:00 AM EST", "task": "US corporate day routine task explanation"},\n'
            '        {"time": "01:00 PM EST", "task": "Collaboration or core engineering focus"},\n'
            '        {"time": "04:00 PM EST", "task": "Standup wrap up or systems validation"}\n'
            '      ],\n'
            '      "skill_gap_steps": [\n'
            '        {"step": "Step 01", "description": "Actionable milestone tailored to US industry standards"},\n'
            '        {"step": "Step 02", "description": "Actionable upskilling step targeting US enterprise stacks"},\n'
            '        {"step": "Step 03", "description": "Final targeted routine to enter the US competitive market"}\n'
            '      ]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        user_context_prompt = (
            f"US Market Profile Inputs:\n"
            f"- Academic Level / Background: {academic_baseline}\n"
            f"- Work-Style Dynamic: {work_style}\n"
            f"- Targeted Functional Strengths: {skills_csv}\n"
            f"- Target US Regional Vector: {geolocation}\n"
            f"- Focus Customization Override: {custom_path if custom_path else 'None Provided'}\n\n"
            f"Calculate the US economic matrices, technical certifications required, and compute standard USD gross compensation brackets."
        )

        # Execute ultra-high-speed LLM completion via active GPT-OSS 20B framework
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_context_prompt}
            ],
            temperature=0.3, 
            max_tokens=4000,
            response_format={"type": "json_object"} 
        )

        raw_response_content = completion.choices[0].message.content
        structured_matrix_data = json.loads(raw_response_content)

        return jsonify(structured_matrix_data)

    except Exception as err:
        # Prints complete stack tracing logs directly to your terminal console
        print("\n-------- [FLASK API RUNTIME ERROR TRACKING LOG] --------")
        traceback.print_exc()
        print("--------------------------------------------------------\n")
        return jsonify({
            "error": "An exception occurred inside the serverless runtime pipeline.",
            "details": str(err)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))