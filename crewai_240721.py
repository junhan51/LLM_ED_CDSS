# https://www.youtube.com/watch?v=-59bKxwir5Q

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

GROQ_API_KEY = os.environ['GROQ_API_KEY']

from crewai import Agent, Task, Crew
from crewai_tools import tool, CSVSearchTool

from langchain_community.tools import DuckDuckGoSearchRun
# pip install --upgrade --quiet  duckduckgo-search

from groq import Groq
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

#*-----------------LLM-----------------*#

llm = ChatGroq(temperature=0.2,
               #format="json",
               model_name="Llama3-70b-8192",
               api_key=os.getenv('GROQ_API_KEY'))



#*-----------------Tools-----------------*#


RxNorm_tool = CSVSearchTool(csv = "./RXNORM_cut.csv", 
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3-70b-8192",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface", # or openai, ollama, ...
            config=dict(
                model="BAAI/bge-small-en-v1.5",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
#pip install sentence-transformers



@tool('DuckDuckGoSearch')
def search_tool(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)


#*-----------------Agents-----------------*#

triage_nurse = Agent(
    role = "Triage Nurse",
    goal = """
        Conduct a thorough and rapid assessment of incoming patients using the KTAS
        (Korean Triage and Acuity Scale) system. Gather comprehensive information about
        patients' symptoms, vital signs, and relevant medical history to determine the
        urgency of care needed and facilitate efficient patient flow in the emergency department.
    """,
    backstory = """
        You are a highly experienced triage nurse with over 15 years of experience in 
        busy emergency departments. Your expertise in rapidly and accurately assessing 
        patient conditions has been crucial in saving countless lives. You have a 
        reputation for remaining calm under pressure and have trained numerous junior 
        nurses in effective triage techniques.
    """,
    tools = [search_tool],
    llm=llm,
    max_iter = 5,
    allow_delegation = False, # controls whether the agent is allowed to delegate tasks to other agents. 'True' is also favorable 
    verbose = True,
)

emergency_physician = Agent(
    role = "Emergency Physician",
    goal = """
        Provide rapid, accurate diagnoses and develop comprehensive treatment plans 
        based on triage information and additional examinations. Make critical decisions 
        about immediate interventions, prescribe appropriate medications, and coordinate 
        with specialists when necessary to ensure optimal patient outcomes.
    """,
    backstory = """
        As a board-certified emergency physician with over a decade of experience, you 
        have handled a wide range of medical emergencies, from major traumas to complex 
        medical conditions. Your ability to make quick, accurate diagnoses and implement 
        effective treatment plans has earned you the respect of your colleagues and the 
        trust of your patients. You are known for your calm demeanor in high-stress 
        situations and your commitment to evidence-based medicine.
    """,
    tools = [search_tool],
    llm=llm,
    max_iter = 5,
    allow_delegation = False, # controls whether the agent is allowed to delegate tasks to other agents. 'True' is also favorable 
    verbose = True,
)

pharmacist = Agent(
    role = "Emergency Room Pharmacist",
    goal = """
        Ensure safe and effective medication use in the emergency room by leveraging 
        your extensive knowledge of pharmacology and the RxNorm database. Review 
        prescriptions meticulously, perform comprehensive drug interaction checks, 
        and provide crucial information about proper medication administration, 
        potential side effects, and dosage adjustments for emergency situations.
    """,
    backstory = """
        You are a highly skilled pharmacist with specialized training in emergency 
        medicine. With over 15 years of experience in hospital pharmacy and emergency 
        departments, your expertise in managing complex medication regimens and 
        preventing adverse drug events is unparalleled. You have contributed to 
        developing hospital-wide protocols for medication safety in emergency 
        situations and are known for your ability to provide rapid, accurate 
        pharmacological consultations in high-pressure environments.
    """,
    tools = [search_tool, RxNorm_tool],
    llm=llm,
    max_iter = 5,
    allow_delegation = False, # controls whether the agent is allowed to delegate tasks to other agents. 'True' is also favorable 
    verbose = True,
)

er_doctor_in_charge = Agent(
    role = "Emergency Room Doctor in Charge",
    goal = """
        Oversee and coordinate all aspects of patient care in the emergency department. 
        Make critical clinical decisions regarding patient management, including 
        treatment plans, admission, and follow-up care. Ensure efficient utilization 
        of resources while maintaining the highest standards of patient care and safety.
    """,
    backstory = """
        With over 20 years of experience in emergency medicine, including 10 years in 
        a leadership role, you are renowned for your clinical acumen and ability to 
        manage complex cases. Your expertise spans across all areas of emergency care, 
        and you have a track record of implementing innovative protocols that have 
        significantly improved patient outcomes and department efficiency. You are 
        respected for your decisive leadership in critical situations and your 
        commitment to mentoring junior staff.
    """,
    tools = [RxNorm_tool, search_tool],
    llm=llm,
    max_iter = 5,
    allow_delegation = False, # controls whether the agent is allowed to delegate tasks to other agents. 'True' is also favorable 
    verbose = True,
)


#*-----------------Tasks-----------------*#

medical_diagnosis = Task(
    description = """
        Based on the triage report and any additional examinations, provide a comprehensive 
        diagnosis and treatment plan for the patient with these symptoms:
        
        <symptoms>
        {input}
        </symptoms>
        
        Your assessment should include:
        1. A thorough analysis of the patient's symptoms, vital signs, and medical history
        2. Differential diagnoses, listing the most likely conditions in order of probability
        3. Recommended diagnostic tests or imaging studies, with justification for each
        4. A detailed initial treatment plan, including medications, procedures, and interventions
        5. Consideration of potential complications and how to mitigate them
        6. Any necessary consultations with specialists, with clear reasons for each referral
        7. A plan for ongoing monitoring and reassessment of the patient's condition

        Use the search tool to find the latest evidence-based guidelines or unusual clinical presentations if necessary.
    """,
    agent = emergency_physician,
    expected_output = """
        Provide a comprehensive medical report including:
        1. Primary working diagnosis with supporting evidence
        2. Differential diagnoses in order of likelihood, with brief explanations
        3. Detailed treatment plan, including:
           a. Medications (dosage, route, frequency) verified with RxNorm
           b. Immediate interventions or procedures
           c. Fluid management if applicable
           d. Pain management strategy
        4. Diagnostic tests ordered, with justification for each
        5. Potential complications to monitor for, with specific warning signs
        6. Consultations requested, if any, with clear rationale
        7. Plan for ongoing monitoring and criteria for reassessment
        8. Relevant evidence-based guidelines or literature referenced (if search tool was used)

        Format your response as follows:
        
        PRIMARY DIAGNOSIS: [State the most likely diagnosis]
        
        Supporting Evidence: [List key findings that support this diagnosis]
        
        Differential Diagnoses:
        1. [Diagnosis 1]: [Brief explanation]
        2. [Diagnosis 2]: [Brief explanation]
        3. [Diagnosis 3]: [Brief explanation]
        
        Treatment Plan:
        1. Medications:
           a. [Drug name, dosage, route, frequency] - [Purpose] (RxNorm verified)
           b. [Drug name, dosage, route, frequency] - [Purpose] (RxNorm verified)
        2. Interventions/Procedures: [List and describe]
        3. Fluid Management: [If applicable]
        4. Pain Management: [Strategy]
        
        Diagnostic Tests:
        1. [Test name]: [Justification]
        2. [Test name]: [Justification]
        
        Potential Complications:
        1. [Complication]: [Warning signs and management plan]
        2. [Complication]: [Warning signs and management plan]
        
        Consultations:
        1. [Specialist]: [Rationale for consultation]
        
        Monitoring Plan: [Describe ongoing monitoring and reassessment criteria]
        
        Evidence-Based Guidelines: [Reference any guidelines or literature used, if applicable]

    """,
)

medication_review = Task(
    description = """
        Conduct a comprehensive review of the prescribed medications for the patient with these symptoms:
        
        <symptoms>
        {input}
        </symptoms>

        Consider their medical history, current condition, and the emergency physician's treatment plan. 
        Your review should include:
        1. A thorough analysis of each prescribed medication using the RxNorm tool
        2. Potential drug interactions, including severity and clinical significance
        3. Dose appropriateness considering the patient's condition, age, weight, and renal/hepatic function
        4. Any contraindications based on the patient's medical history or current condition
        5. Identification of any high-alert medications that require special handling or monitoring
        6. Recommendations for medication adjustments, alternatives, or additional monitoring if necessary
        7. Important side effects or adverse reactions to watch for in the emergency setting
        8. Specific administration instructions or precautions for the nursing staff
        9. Any drug-disease interactions that could impact the patient's current condition

        Use the RxNorm tool extensively to gather detailed information about each medication. 
        Use the search tool to find the latest pharmacological guidelines or information on rare 
        drug effects if necessary.
    """,
    agent = pharmacist,
    expected_output = """
        Provide a detailed medication safety report including:
        1. List of prescribed medications with a comprehensive analysis of each
        2. Identified drug interactions, contraindications, or concerns
        3. Dose appropriateness evaluations and any recommended adjustments
        4. Specific administration instructions and precautions
        5. Potential adverse effects to monitor in the emergency setting
        6. Recommendations for medication therapy optimization
        7. Any additional pharmacological considerations relevant to the patient's care

        Format your response as follows:
        
        MEDICATION SAFETY REPORT
        
        Patient Condition Summary: [Brief overview of relevant clinical information]
        
        Prescribed Medications Analysis:
        1. [Drug Name] (RxNorm verified):
           - Dose/Route/Frequency: [As prescribed]
           - Indication: [For what condition]
           - Appropriateness: [Comment on dose, considering patient factors]
           - Interactions: [List any significant interactions]
           - Contraindications: [If any, based on patient's condition]
           - Administration Instructions: [Specific guidance for nursing]
           - Monitoring: [What to watch for, including adverse effects]
        
        2. [Repeat for each medication]
        
        Overall Medication Therapy Assessment:
        - Drug-Disease Interactions: [Any concerns with patient's conditions]
        - High-Alert Medications: [Identify any that require special precautions]
        - Pharmacokinetic Considerations: [Any adjustments needed for renal/hepatic function]
        
        Recommendations:
        1. [Specific recommendation for medication adjustment, monitoring, or alternative]
        2. [Additional recommendations as needed]
        
        Emergency Pharmacology Considerations:
        - [Any special considerations for medication use in the ER setting]
        
        References:
        - [List any guidelines or resources used from the search tool, if applicable]
    """,
)

triage_assessment = Task(
    description = """
        Conduct a comprehensive triage assessment of the patient presenting with these symptoms:

        <symptoms>
        {input}
        </symptoms> 
        
        Utilize the KTAS system to determine the urgency of care needed. Your assessment should include:
        1. A detailed evaluation of the patient's current symptoms and their severity
        2. Complete set of vital signs (blood pressure, heart rate, respiratory rate, temperature, oxygen saturation)
        3. Relevant medical history, including chronic conditions, allergies, and current medications
        4. Any recent trauma or significant events related to the current condition
        5. Pain assessment using a standardized scale
        6. Mental status evaluation
        7. Any immediate life-threatening conditions that require urgent intervention

        Refer to the KTAS (Korean Triage and Acuity Scale) guidelines below.
        <KTAS_guide>
        1: 
            description: "Conditions that require immediate intervention; life-threatening or potentially life-threatening states (or high risk of rapid deterioration)",
            examples: ["Cardiac arrest", "Respiratory arrest", "Unconsciousness not related to alcohol consumption"],
            priority: "Highest priority"
        
        2: 
            description: "Conditions with potential threats to life, limb, or organ function requiring rapid medical intervention",
            examples: ["Myocardial infarction", "Cerebral hemorrhage", "Cerebral infarction"],
            priority: "Second priority"
        
        3: 
            description: "Conditions that may progress to a serious problem requiring emergency intervention",
            examples: ["Dyspnea (with oxygen saturation above 90%)", "Diarrhea with bleeding"],
            priority: "Third priority"
        
        4: 
            description: "Conditions that, considering the patient's age, pain level, or potential for deterioration or complications, could be treated or re-evaluated within 1-2 hours",
            examples: ["Gastroenteritis with fever above 38°C", "Urinary tract infection with abdominal pain"],
            priority: "Fourth priority"
        
        5: 
            description: "Conditions that are urgent but not emergencies, or those resulting from chronic problems with low risk of deterioration",
            examples: ["Common cold", "Gastroenteritis", "Diarrhea", "Laceration (wound)"],
            priority: "Fifth priority"
        
        </KTAS_guide>

        Use the search tool to find any additional information about unusual symptoms or conditions if necessary.
    """,
    agent = triage_nurse,
    expected_output = """
        Provide a comprehensive triage report including:
        1. KTAS level assigned (1-5) with detailed justification based on the KTAS criteria
        2. Comprehensive summary of symptoms, vital signs, and relevant medical history
        3. List of current medications and allergies, verified using the RxNorm tool
        4. Any critical or unusual findings that require immediate attention
        5. Recommended immediate actions or precautions for the medical team
        6. Any additional information gathered from the search tool, if used

        Format your response as follows:
        
        KTAS CLASSIFICATION: [Level]
        
        Detailed Justification: [Explain why this KTAS level was assigned]
        
        Patient Assessment:
        1. Presenting Symptoms: [List and describe]
        2. Vital Signs: [List all measured vital signs]
        3. Medical History: [Relevant past and current conditions]
        4. Medications and Allergies: [List with RxNorm verifications]
        5. Pain Assessment: [Score and description]
        6. Mental Status: [Brief evaluation]
        
        Critical Findings: [Any immediate life-threatening conditions]
        
        Recommended Actions: [Immediate steps for the medical team]
        
        Additional Information: [Any relevant data from search tool, if used]
    """,
    context = [
    medical_diagnosis,
    medication_review,
    ],
)

er_management_decision = Task(
    description = """
        As the Emergency Room Doctor in Charge, review all the information provided by the triage nurse, 
        emergency physician, and pharmacist for the patient with these symptoms:
        
        <symptoms>
        {input}
        </symptoms> 
        
        Based on this comprehensive 
        assessment, make critical clinical decisions regarding the patient's care. Your decision-making 
        process should include:
        1. A thorough review of the KTAS classification and its implications for immediate care
        2. Evaluation of the diagnosis, differential diagnoses, and proposed treatment plan
        3. Consideration of the medication safety report and any pharmacological concerns
        4. Assessment of the need for immediate interventions, further diagnostic tests, or specialist consultations
        5. Determination of the most appropriate next steps for patient care (e.g., continued ER management, 
           admission to a specific unit, transfer to a specialized facility, or safe discharge with follow-up)
        6. Consideration of resource utilization and department capacity in your decision-making
        7. Development of a clear, actionable plan for ongoing patient care and monitoring

        Use the RxNorm tool to double-check any medication decisions. Use the search tool to find relevant 
        clinical guidelines or hospital protocols if necessary for decision-making.
    """,
    agent = er_doctor_in_charge,
    expected_output = """
        Provide a comprehensive management decision including:
        1. Restatement of the KTAS classification with your assessment of its accuracy
        2. Your agreement or adjustments to the primary diagnosis and treatment plan
        3. Critical decision on patient disposition (continue ER care, admit, transfer, or discharge)
        4. Detailed justification for your decision, including clinical and logistical factors
        5. Specific instructions for next steps in patient care, including any changes to the treatment plan
        6. Additional resources, specialists, or interventions needed
        7. Clear communication plan for the patient, family, and other healthcare providers
        8. Contingency plans or criteria for reassessing the patient's condition

        Format your response as follows:
        
        EMERGENCY DEPARTMENT MANAGEMENT DECISION
        
        KTAS Classification Review: [Restate and assess accuracy]
        
        Clinical Assessment:
        - Primary Diagnosis: [State agreement or provide alternative with justification]
        - Critical Findings: [Highlight key issues requiring immediate attention]
        
        Disposition Decision: [State clearly: Continue ER care / Admit / Transfer / Discharge]
        
        Justification:
        [Provide a detailed explanation for your decision, referencing clinical findings, 
        resource considerations, and best practices]
        
        Management Plan:
        1. Immediate Actions: [List any immediate interventions required]
        2. Medications: [Any changes or confirmations to the proposed medication plan]
        3. Diagnostic Tests: [Additional tests ordered or results to be followed up]
        4. Consultations: [Any specialist consultations required, with urgency level]
        5. Monitoring: [Specific parameters to be monitored and frequency]
        
        Resource Allocation:
        [Specify any special resources needed and how they will be utilized]
        
        Communication Plan:
        - Patient/Family: [Key points to be communicated]
        - Healthcare Team: [Instructions for ER staff, admitting team, or follow-up providers]
        
        Contingency Planning:
        [Criteria for reassessment or change in management plan]
        
        Additional Considerations:
        [Any other relevant factors influencing your decision]
        
        References:
        [List any guidelines or protocols referenced, if search tool was used]
    """,
    context = [
        triage_assessment,
        medical_diagnosis,
        medication_review,
    ],
    output_file = "er_management_decision.md",
)



#*-----------------Crew-----------------*#

class EmergencyRoomQA:
    def __init__(self):
        self.crew = Crew(
            tasks=[
                medical_diagnosis,
                medication_review,
                triage_assessment,
                er_management_decision
            ],
            agents=[
                emergency_physician,
                pharmacist,
                triage_nurse,
                er_doctor_in_charge
            ],
            verbose=2,
        )

    def get_result(self, symptoms):
        result = self.crew.kickoff(
            inputs={
                "input": symptoms
            }
        )
        return result
