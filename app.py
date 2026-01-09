from rag import rag
import streamlit as st

st.title("Outil de recherche de connaissances - Compatibilité de médicaments ou d'un médicament et d'une maladie")
medication_1 = st.text_input("Medication 1")
medication_2 = st.text_input("Medication 2")
if st.button("Generate"):
    if not medication_1 or not medication_2:
        st.write("Please enter both medications.")
    else:
        query = medication_1 + " and " + medication_2
        response, retrieval, evidence = rag(query)
        highlighted_evidence = retrieval.replace(evidence, f"<mark style='background-color:#ffe066; padding:2px 4px; border-radius:3px;'>{evidence}</mark>")
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
                            <strong>📄 Documents sources :</strong><br>
                            {highlighted_evidence.replace('\n','<br>')}
                        </div>
                        """, unsafe_allow_html=True)

