# Disaster Response Pipeline Project

## Summary


## File structure


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
