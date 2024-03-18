﻿# Run Book and Other details
1) Clone the Project
2) Make sure python is installed and PATH vaiable is set appropriately
3) To run the Project in local run following commands
   a) pip install -r requirements.txt
   b) uvicorn main:app --reload
4) Make any code changes required
5) Rise the PR with git
6) Upon merge CICD is automatically kicked and the application will be deployed to CLoud Azure Web App
7) To test the App:
    Postman URL: https://identifyhumanvoice.azurewebsites.net/voice/analyse/
    Click on Body -> select Form Data -> Key: upload_file and in Drop Down select File. Attach file in Value and hit send
8) The PING url is https://identifyhumanvoice.azurewebsites.net/voice/analyse/ping .  It should return "Health Check Sucessful"
