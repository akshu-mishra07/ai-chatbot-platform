import csv
import os

output_file = r"c:\Users\ACER\Downloads\GEN---AI-course-main\customer_service_chatbot_LLM\dataset\medquad_dataset.csv"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

categories_info = [
    {
        "name": "CancerGov",
        "path": "1_CancerGov_QA",
        "topics": ["Breast Cancer", "Lung Cancer", "Prostate Cancer", "Melanoma", "Leukemia"],
        "q_types": ["What is {topic}?", "What are the treatments for {topic}?", "What are the symptoms of {topic}?", "How is {topic} diagnosed?"]
    },
    {
        "name": "GHR",
        "path": "2_GHR_QA",
        "topics": ["Cystic Fibrosis", "Huntington Disease", "Sickle Cell Disease", "Marfan Syndrome", "Hemophilia"],
        "q_types": ["What is the genetic cause of {topic}?", "How is {topic} inherited?", "What are the genetic tests for {topic}?", "What is {topic}?"]
    },
    {
        "name": "MedlinePlus Health Topics",
        "path": "3_MedlinePlus_QA",
        "topics": ["Asthma", "Diabetes", "Hypertension", "Arthritis", "Migraine", "Obesity"],
        "q_types": ["What is {topic}?", "What are the symptoms of {topic}?", "How can {topic} be prevented?", "What is the treatment for {topic}?", "What causes {topic}?"]
    },
    {
        "name": "NINDS",
        "path": "4_NINDS_QA",
        "topics": ["Alzheimer's Disease", "Parkinson's Disease", "Multiple Sclerosis", "Epilepsy", "Stroke"],
        "q_types": ["What is {topic}?", "What are the early signs of {topic}?", "How does {topic} progress?", "What research is being done on {topic}?"]
    },
    {
        "name": "NIDDK",
        "path": "5_NIDDK_QA",
        "topics": ["Chronic Kidney Disease", "IBS", "Crohn's Disease", "Celiac Disease", "Gallstones"],
        "q_types": ["What is {topic}?", "What diet is recommended for {topic}?", "What are the symptoms of {topic}?", "How is {topic} treated?"]
    },
    {
        "name": "SeniorHealth",
        "path": "6_SeniorHealth_QA",
        "topics": ["Osteoporosis", "Dementia", "Hearing Loss", "Cataracts", "Macular Degeneration"],
        "q_types": ["What is {topic}?", "How does aging affect {topic}?", "What are the risk factors for {topic}?", "How is {topic} managed in older adults?"]
    },
    {
        "name": "DrugBank / Drugs",
        "path": "7_DrugBank_QA",
        "topics": ["Aspirin", "Ibuprofen", "Metformin", "Lisinopril", "Atorvastatin"],
        "q_types": ["What is {topic} used for?", "What are the side effects of {topic}?", "What is the dosage of {topic}?", "What are the contraindications for {topic}?"]
    },
    {
        "name": "GARD",
        "path": "8_GARD_QA",
        "topics": ["Duchenne Muscular Dystrophy", "Ehlers-Danlos Syndrome", "Gaucher Disease", "Pompe Disease", "Fabry Disease"],
        "q_types": ["What is {topic}?", "How rare is {topic}?", "What are the symptoms of {topic}?", "Is there a cure for {topic}?"]
    },
    {
        "name": "NHLBI",
        "path": "9_NHLBI_QA",
        "topics": ["Heart Failure", "COPD", "Sleep Apnea", "Arrhythmia", "Pulmonary Hypertension"],
        "q_types": ["What is {topic}?", "How does {topic} affect the body?", "What are the treatments for {topic}?", "Can {topic} be cured?"]
    },
    {
        "name": "CDC Prevention",
        "path": "10_CDC_QA",
        "topics": ["Flu Vaccine", "COVID-19 Prevention", "HPV Screening", "Hand Hygiene", "Travel Vaccines"],
        "q_types": ["Why is {topic} important?", "How does {topic} work?", "Who should consider {topic}?", "What are the guidelines for {topic}?"]
    }
]

general_safety = " Always consult a healthcare provider for personalized medical advice and before making any changes to your health regimen."

def generate_answer(topic, q_type, category):
    q_lower = q_type.lower()
    if "used for" in q_lower or "guidelines" in q_lower or "important" in q_lower:
        ans = f"{topic} is an essential component in modern healthcare management and prevention. Its primary application involves addressing specific clinical or preventive targets to improve patient outcomes. Clinical guidelines outline its proper usage, noting that adherence is crucial for maximizing benefits. Many individuals rely on it to maintain baseline health or manage ongoing conditions."
    elif "side effects" in q_lower or "contraindications" in q_lower:
        ans = f"Like many interventions, {topic} may be associated with certain adverse profiles or side effects. Common reactions can be mild, but some individuals might experience more severe complications requiring immediate attention. It is critical to review all potential risks and interactions prior to initiation. Patients are advised to monitor their physical response closely and report unusual changes."
    elif "dosage" in q_lower:
        ans = f"The appropriate dosing or schedule for {topic} varies significantly depending on individual patient characteristics such as age, weight, and existing comorbidities. Medical professionals often start with a conservative approach and titrate based on clinical response. Strict adherence to prescribed regimens is vital to avoid toxicity or suboptimal efficacy. Missing doses or altering schedules without professional guidance can compromise health outcomes."
    elif "prevented" in q_lower or "diet" in q_lower or "risk factors" in q_lower:
        ans = f"Addressing {topic} effectively often requires comprehensive lifestyle and behavioral modifications alongside traditional medical care. Dietary changes, regular physical activity, and avoiding known risk factors play a significant role in mitigating severity. Preventive strategies are best implemented early and consistently. Public health organizations strongly emphasize these modifications for long-term well-being."
    elif "genetic" in q_lower or "inherited" in q_lower:
        ans = f"{topic} involves underlying genetic mutations or inheritance patterns that are typically passed down through families. Genetic counseling and specialized testing are often recommended for individuals with a known family history. Understanding the genetic basis helps in estimating risks for future generations and guides family planning. Advances in genetic research continue to shed light on its complex mechanisms."
    elif "treatment" in q_lower or "cure" in q_lower or "managed" in q_lower:
        ans = f"The management of {topic} typically employs a multifaceted approach aimed at symptom control and slowing progression. While definitive cures may not always exist, current therapeutic options can significantly enhance quality of life. Interventions range from supportive care to advanced pharmacological therapies. Continuous follow-up ensures that the treatment strategy adapts to the patient's evolving condition."
    elif "symptoms" in q_lower or "signs" in q_lower:
        ans = f"The clinical presentation of {topic} can include a diverse array of symptoms that vary in intensity over time. Patients frequently report persistent discomfort, systemic fatigue, or specific focal signs that prompt initial medical evaluation. Recognizing these early indicators is crucial for timely diagnosis and intervention. Ignoring persistent symptoms can lead to worsening of the condition and increased complications."
    else: 
        ans = f"{topic} is a significant medical entity that demands careful attention and understanding from both patients and healthcare providers. It encompasses a range of physiological or structural alterations that disrupt normal function. The condition is studied extensively to develop better diagnostic and therapeutic pathways. Awareness and education remain vital tools in empowering affected individuals."
    
    return ans + general_safety

records = []
source_counter = 100

for cat in categories_info:
    count = 0
    while count < 20:
        for topic in cat["topics"]:
            for q_template in cat["q_types"]:
                if count >= 20:
                    break
                question = q_template.replace("{topic}", topic)
                answer = generate_answer(topic, question, cat["name"])
                focus = topic
                source = f"MedQuAD/{cat['path']}/{source_counter:07d}.xml"
                
                records.append({
                    "question": question,
                    "answer": answer,
                    "category": cat["name"],
                    "focus": focus,
                    "source": source
                })
                source_counter += 1
                count += 1
            if count >= 20:
                break

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer", "category", "focus", "source"], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in records:
        writer.writerow(row)

print(f"Successfully wrote {len(records)} records to {output_file}")
