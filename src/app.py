import streamlit as st
from model_one import model1
from model_two import model2
from model_three import model3


def main():
    """Display of the capstone project."""

    st.title("DEV - SUBJECT TO CHANGE") # TODO: remove before final production version.
    st.title("Capstone Project Team JIM")
    st.write("Authors: Jenny Ney, Melanie Starr, Ian Byrne")

    menu = ["Home", "Tree Analysis", "VAR Analysis", "RNN Analysis", "Conclusion"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("About page")
        st.write(
            """With this app the authors indend to the user to be able to follow along
            with design choices made in each model and view our results."""
        )

    if choice == "Tree Analysis":
        st.subheader("Tree Analysis")
        model1()

    if choice == "VAR Analysis":
        st.subheader("VAR Anaylsis")
        model2()

    if choice == "RNN Analysis":
        st.subheader("Neural Network Analysis")
        model3()

    if choice == "Conclusion":
        st.subheader("Conclusions")
        st.write("Summary of Capstone project conlusions.")


if __name__ == "__main__":
    main()
