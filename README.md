# Disaster Response Pipeline Project

## Summary
The DRP project is based on communications in disaster situations.  Because response time in these situations can mean the difference between life and death, it is important that these messages are noticed by the relevant parties as soon as possible, without irrelevant messages getting in the way, and for that they need to be categorized accordingly.

Figure Eight has collated a large database of emergency messages, as well as the categories they fall into.  The purpose of the program is to process this data, analyze it, and use it to build a model that can categorize new emergency messages automatically.  The results are then showcased in a web app.

## File structure
* App
	* run.py - Code to start the app
	* templates
		* go.html - Classificaiton result page of web app
		* master.html - Main page of web app
* Data
	* disaster_categories.csv - Category data from Figure Eight
	* disaster_messages.csv - Message data from Figure Eight
	* DisasterResponse.db (optional) - Where the processed data is stored
	* process_data.py - Code to load and process the data
* Models
	* classifier.pkl (optional) - Where the trained model is stored
	* train_classifier.py - Code to create and train a categorizing model
* README.md - You are here!

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. If you are not running from the Project Workspace IDE, go to http://0.0.0.0:3001/ in your web browser.

### If you are running from the Project Workspace IDE, follow steps 1 and 2 as above, and then do the following:
3. Open a new Terminal tab and run:
	`env|grep WORK`
	You will see a SPACEID and SPACEDOMAIN.  Make a note of them both.

4. Open a new browser tab and go to:
	https://SPACEID-3001.SPACEDOMAIN
    ...with SPACEID and SPACEDOMAIN replaced with what you saw in the previous step.
    
5. If the above does not work, you may have to download the project to a local computer and run it there.
