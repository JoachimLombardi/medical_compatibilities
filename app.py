import json
from rag import rag
import streamlit as st


st.title("Compatibilité de médicaments ou d'un médicament et d'une maladie")
treatment = st.text_area("Traitement")
if st.button("Generate"):
    if not treatment:
        st.write("Veuillez entrer un traitement.")
    else:
        query = treatment 
        response, adverse_effects, evidence_compatibility = rag(query)
        merged = {}
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
        for effect in adverse_effects:
            medication_name = effect.get("medication_name", "")
            merged.setdefault(medication_name, set()).add(effect.get("evidence", ""))
        for evidence in evidence_compatibility:
            medication_name = evidence.get("medication_name", "")
            merged.setdefault(medication_name, set()).add(evidence.get("evidence", ""))
        with open("data/json/notices.json", "r") as f:
            notices = json.load(f)
        for notice in notices:
            highlighted_evidence = notice["text"]
            for snippet in merged.get(notice["medication_name"], []):
                if snippet in highlighted_evidence:
                    highlighted_evidence = highlighted_evidence.replace(snippet, 
                                                                       f"""<mark style=
                                                                       'background-color:#ffe066; 
                                                                       padding:2px 4px; 
                                                                       border-radius:3px;'>{snippet}</mark>""")
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
                                <strong>📄 Documents sources : {notice["medication_name"]}</strong><br>
                                {highlighted_evidence.replace('\n','<br>')}
                            </div>
                            """, unsafe_allow_html=True)

