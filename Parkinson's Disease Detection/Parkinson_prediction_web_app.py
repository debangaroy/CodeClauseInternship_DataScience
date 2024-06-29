import numpy as np
import streamlit as st
import pickle

# Load the trained model and scaler
model_path = "O:/virtual_intern/CodeClause-Data Science/Parkinson's Disease Detection/trained_classifier_model.sav"
scaler_path = "O:/virtual_intern/CodeClause-Data Science/Parkinson's Disease Detection/scaler.sav"
loaded_model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# Prediction function
def parkinson_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_input_data = scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_input_data)
    if prediction[0] == 0:
        return "The person doesn't have Parkinson's disease"
    else:
        return "The person has Parkinson's disease"

# Streamlit app
def main():
    st.title("Parkinson's Disease Prediction Web App")

    # Getting input data from the user
    MDVP_Fo_Hz = st.text_input("MDVP Fo(Hz)")
    MDVP_Fhi_Hz = st.text_input("MDVP Fhi(Hz)")
    MDVP_Flo_Hz = st.text_input("MDVP Flo(Hz)")
    MDVP_Jitter_percent = st.text_input("MDVP Jitter(%)")
    MDVP_Jitter_Abs = st.text_input("MDVP Jitter(Abs)")
    MDVP_RAP = st.text_input("MDVP RAP")
    MDVP_PPQ = st.text_input("MDVP PPQ")
    Jitter_DDP = st.text_input("Jitter DDP")
    MDVP_Shimmer = st.text_input("MDVP Shimmer")
    MDVP_Shimmer_dB = st.text_input("MDVP Shimmer(dB)")
    Shimmer_APQ3 = st.text_input("Shimmer APQ3")
    Shimmer_APQ5 = st.text_input("ShimmerAPQ5")
    MDVP_APQ = st.text_input("MDVP APQ")
    Shimmer_DDA = st.text_input("Shimmer DDA")
    NHR = st.text_input("NHR")
    HNR = st.text_input("HNR")
    RPDE = st.text_input("RPDE")
    DFA = st.text_input("DFA")
    spread1 = st.text_input("spread1")
    spread2 = st.text_input("spread2")
    D2 = st.text_input("D2")
    PPE = st.text_input("PPE")

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button("Prediction Test Result"):
        diagnosis = parkinson_prediction([MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, 
                                          MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, 
                                          MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, 
                                          Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])

    st.success(diagnosis)

if __name__ == "__main__":
    main()
