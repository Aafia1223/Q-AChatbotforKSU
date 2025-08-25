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

1- Download Webscrape.py, chatbot.py and  main.py and put them in the same directory.

2- Download menu_hierarchy.json and put them anywhere in any folder or even in the same directory as the Python files. 

3- Import necessary libraries from requirements.txt.

4- In main.py, in the Config section, put the path of the menu_hierarchy.json to JSON_FILE_PATH.

5- Run main.py by following these instructions:

    - Ensure the terminal directs to the directory these files are stored in
    
    - To run main.py locally, type this in its terminal: uvicorn main:app --host 0.0.0.0 --port 8000
    
    - Open this on the browser to see the result: http://127.0.0.1:8000/

Updates will be made soon
