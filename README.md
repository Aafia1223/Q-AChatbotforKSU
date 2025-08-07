# Q-A Chatbot for KSU

This is a chatbot that answers questions related to King Saud University. 
The chatbot answers questions about:
- Admission Information (requirements and processes)
- Academic Calendar
- College categories, their colleges and their department information (Faculty directories and contact information about departments)
- Student services (Housing, IT Helpdesk, libraries)
- Policies and regulations (Plagiarism, grading and etc)
- FAQs
- Research institutes

## How to open the chatbot 

1- Download Webscrape.py, chatbot.py, app.py and  main.py and put them in the same directory.

2- Download menu_hierarchy.json and put them anywhere in any folder or even in the same directory as the Python files. 

3- Import necessary libraries from requirements.txt.

4- In main.py, in the Config section, put the path of the menu_hierarchy.json to JSON_FILE_PATH.

5- First run main.py and then app.py in different terminals. To run them:

    - Ensure both terminals direct to the folder main.py and app.py are in
    
    - To run main.py, type this in its terminal: -m uvicorn main:app --reload
    
    - To run app.py, type this in its terminal: streamlit run app.py
