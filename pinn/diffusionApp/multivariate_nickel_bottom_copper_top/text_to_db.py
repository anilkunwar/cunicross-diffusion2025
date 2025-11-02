import streamlit as st
import sqlite3
import os

def create_db(text, db_path):
    # Connect to SQLite database (it will create if not exists)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create a table for experiment descriptions (if not exists)
    c.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT
        )
    ''')
    # Insert uploaded text as a new record
    c.execute('INSERT INTO experiments (description) VALUES (?)', (text,))
    conn.commit()
    conn.close()

def main():
    st.title("Experiment Description to DB")

    uploaded_file = st.file_uploader("Upload your experiment description text file", type=['txt'])
    if uploaded_file is not None:
        # Read uploaded file content as string
        text = uploaded_file.getvalue().decode("utf-8")
        st.text_area("Preview of uploaded content", text, height=200)

        if st.button("Create DB file"):
            db_filename = "description_of_experiment.db"
            create_db(text, db_filename)
            st.success(f"Database file '{db_filename}' created successfully!")

            with open(db_filename, "rb") as file:
                btn = st.download_button(
                    label="Download database file",
                    data=file,
                    file_name=db_filename,
                    mime="application/x-sqlite3"
                )
            # Optional: remove db file after download button (comment if you want to keep)
            # os.remove(db_filename)

if __name__ == "__main__":
    main()
