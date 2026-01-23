# docker run -p 6333:6333 qdrant/qdrant
import json
from rag import rag
import streamlit as st
from ocr import call_llm


st.title("Compatibilité de médicaments ou d'un médicament et d'une maladie")
mode = st.radio(
    "Avez-vous un traitement en cours ?",
    ["❎ Non", "✍️ Saisie manuelle", "📄 Charger un document"]
)
if mode == "✍️ Saisie manuelle":
    old_treatments = st.text_area("Traitement en cours")
elif mode == "📄 Charger un document":
    old_treatments = st.file_uploader("Ancien traitement", 
                                      type=["jpg", "jpeg", "png", "pdf"],
                                      accept_multiple_files=True)
    old_treatments = call_llm(old_treatments)
else:
    old_treatments = ""
treatments = st.file_uploader(
    "Traitement",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True)
if st.button("Generate"):
    if not treatments:
        st.write("Veuillez télécharger votre nouveau traitement")
    else:
        # treatment = old_treatments + "\n" + call_llm(treatments)
        treatment = old_treatments + "\n" + """ERLEADA 240 MG CPR (APALUTAMIDE)
        DECAPEPTYL LP 11,25MG PDR ET SOL INJ (TRIPTORELINE PAMOATE)
        CRESTOR 20MG CPR (ROSUVASTATINE)
        RESITUNE 75 MG CPR (ACIDE ACETYLSALICYLIQUE)
        LOXEN LP 50MG GELULE (NICARDIPINE)"""
        response, evidence_compatibility = rag(treatment)
        print(evidence_compatibility)
        adverse_effects = {}
        adverse_effect = ""
        with st.container():
            st.markdown(f"""
                            <div style="
                                border: 2px solid green; 
                                border-radius: 10px; 
                                padding: 15px; 
                                background-color: #f0fff0;
                                margin-bottom: 20px;
                            ">
                            <strong>✅ Réponse :</strong><br>
                            {response.replace('\n','<br>')}
                            </div>
                            """, unsafe_allow_html=True)
        with open("data/json/points.json", "r") as f:
            points = json.load(f)
        with open("data/json/notices.json", "r") as f:
            notices = json.load(f)
        for notice in notices:
            highlighted_evidence = notice["text"]
            print(repr(highlighted_evidence))
            for point in points:
                if point.get("payload").get("adverse_effects"):
                    snippet = point.get("payload").get("text")
                    if snippet and snippet.strip() and snippet in highlighted_evidence:
                        highlighted_evidence = highlighted_evidence.replace(snippet, 
                                                                        f"""<mark style='background-color:#f0fff0; padding:2px 4px; border-radius:3px;'>{snippet}</mark>""")
            for snippet in evidence_compatibility:
                    if snippet and snippet.strip() and snippet in highlighted_evidence:
                        highlighted_evidence = highlighted_evidence.replace(snippet, 
                                                                        f"""<mark style='background-color:#ffe066; padding:2px 4px; border-radius:3px;'>{snippet}</mark>""")
            with st.container():
                st.markdown(f"""
                            <div style="
                                border: 2px solid blue; 
                                border-radius: 10px; 
                                padding: 15px; 
                                background-color: #f0f8ff;
                                max-height: 300px;
                                overflow-y: auto;
                            ">
                                <strong>📄 Document source : Notice {notice["medication_name"]}</strong><br>
                                {highlighted_evidence.replace('\n','<br>')}
                            </div>
                            """, unsafe_allow_html=True)

