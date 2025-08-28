# Libraries for PyQt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, # Foundational
							QMessageBox, QFileDialog, QDialog, # Pop ups 
							QVBoxLayout, QGridLayout, QScrollArea, # Layouts
							QTableWidget, QTableWidgetItem, # Table-related
							QStyledItemDelegate, # Delegate for date in table
							QLabel, QLineEdit, QDateEdit, QComboBox, # Other general widgets
							QCheckBox, QPushButton, QRadioButton, QButtonGroup) 
from PyQt6.QtCore import Qt, QDate, QTimer
from PyQt6.QtGui import QDoubleValidator

# Libraries for ML
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import joblib # to save ML model to pkl file

# Libraries for charting
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

# Other Libraries
import sys # for argv and exit
import os # for checking if file is exist
import mysql.connector # for database queries
import json # for parsing json
from datetime import datetime # for converting datetime formats

# User database configuration
config = {
		"database": "commissionWorkAccountingDB",
		"user": "user1",
		"password": "user1password",
		"host": "localhost",
		"port": 3306,
		"charset": "utf8mb4",
		"collation": "utf8mb4_general_ci"
	}

def error(title, msg, terminate=False):
	"""
	Displays an error message to the user. 
	Defined globally to be shared across all classes.

	Args: 
		title (str): Title of the error.
		msg (str): Error message to display to the user.
		terminate (boolean): if True, terminate program. Default is False.
	"""
	# icon is critical if terminating, otherwise just a warning
	icon = QMessageBox.Icon.Critical if terminate else QMessageBox.Icon.Warning

	msgBox = QMessageBox()
	msgBox.setIcon(icon)
	msgBox.setWindowTitle(title)
	msgBox.setText(msg)
	msgBox.exec()

	if terminate:
		sys.exit(1)

class ML_MODEL:
	"""
	Machine Learning Class used to predict the pay based on the Job type and 
	hours worked. 

	This class uses:
	- The decision tree model to predict.
	- Implements non-volatile memory to keep track of changes in the database 
		and previously trained ML.
	- Implements a cache system for efficient data prediction.
	"""

	# ****************************** CONSTRUCTOR ******************************

	def __init__(self, dbConn):
		# ------------------------ Field Declarations ------------------------

		self.dbConn = dbConn # database sql connection
		self.model = None # ML Model, to be initialized in predict method
		# cache predictions to improve efficiency by not predicting everytime
		self.predictionCache = {} # key (int): ID, value (float): predicted pay

		# ----------------------------- Filepaths -----------------------------

		# File that keeps track of changes in the db
		self.modelConfigFilepath = "./modelConfig.json"
		# ML model file
		self.modelPklFilepath = "./model.pkl"
	
	# ************************** MAIN PUBLIC METHODS **************************

	def predict(self, requestedIDs):
		"""
		Predicts the pay of requested IDs and return them as 
		a dict of IDs (int) and Predicted Pay (float).

		The design of this class calls for more efficient way to provide 
		predictions since it will be called mutiple times when rendering the 
		table. Thus, this utilize a cache system, represented as a 
		dict of IDs (int) and Predicted Pay (float).

		There are 3 cases when this method is called (ordered by priority):
		Case 1: retrainLoadFactor > 20%
			If 20% or more of the entries has been altered, this will trigger 
			an ML retraining to keep predictions accurate. Then, it will predict 
			everything in the database. The predictions will be cached and the 
			requested IDs will be returned.
		Case 2: First time starting the program or after reset()
			This will either load the ML model from file if available, 
			otherwise do a complete ML training from the database. Then, it will 
			predict everything in the database. The predictions will be cached 
			and the requested IDs will be returned.
		Case 3: Predict only the dirty IDs (IDs that has been altered)
			This is the normal operation of this method. It will predict only
			the dirty IDs in the requestedIDs, then cache them. All requestedIDs
			will be returned.
			
		Args:
			requestedIDs (list): List of IDs to be predicted for pay.
		
		Returns:
			Predicted Pay as a dict of IDs (int) and Predicted Pay (float).
		"""

		# --------------------------- Guard Clause ---------------------------

		# Do not predict an empty dataset
		if len(requestedIDs) == 0:
			return {}
		
		# ------------------------------ Case 1 ------------------------------ 
		# ------------- If retrain load factor is above threshold -------------
		if (self._calculateRetrainLoadFactor() > 0.2):
			# print("numOfChanges is above threshold, retraining")
			self._train()
			self._predictAndCacheEverything()
		# ------------------------------ Case 2 ------------------------------ 
		# ---------------- First time starting up the program ----------------
		elif self.model == None:
			# print("First time loading up the ML")
			# Either load the model from pkl file if exist or train from scratch
			if os.path.exists(self.modelPklFilepath):
				# print("loading model from file")
				self.model = joblib.load(self.modelPklFilepath)
			else:
				# print("training model")
				self._train()

			# predict and cache everything
			self._predictAndCacheEverything()
		# ------------------------------ Case 3 ------------------------------ 
		# -------------------- Predict only the dirty IDs --------------------
		else:
			dirtyIDs = list(filter(lambda x: x not in self.predictionCache, requestedIDs))
			if len(dirtyIDs) != 0:
				# print("predicting and caching dirty IDs only")
				self._predictAndCacheDirtyIDs(dirtyIDs)
			# else:
			# 	print("Cache hit!!!")

		# ------------------ Shared code between all 3 cases ------------------
		# ----------------------- return the asked ids -----------------------
		predictionsToReturn = {}
		for id in requestedIDs:
			predictionsToReturn[id] = self.predictionCache[id]

		# print("returning requested ML predictions\n")
		return predictionsToReturn
	
	def incrementDbChangeCounter(self, x = 1):
		"""
		Increments the dbChangeCounter in the self.modelConfigFilepath x times 
		(default is 1). This should be called if feature or label is modified, 
		inserted, or deleted.

		x (int): The number of times to increment changes.
		"""

		config = self._readModelConfigFilepath()
		config["dbChangeCounter"] += x
		self._writeModelConfigFilepath(config)
	
	def markDirtyIDs(self, IDs):
		"""
		Marks IDs as dirty by removing it from the cache (self.predictionCache).

		IDs (list): IDs to be marked dirty (removed from self.predictionCache).
		"""

		for id in IDs:
			self.predictionCache.pop(id, None)

	def reset(self):
		"""
		Remove all history and training of the ML model in memory and disk. 
		This method is used when the database is cleared.
		"""

		# Reset in memory: set self.model to None, clear self.predictionCache
		self.model = None
		self.predictionCache.clear()

		# Reset in disk: remove modelConfigFilepath and modelPklFilepath files
		if os.path.exists(self.modelConfigFilepath):
			os.remove(self.modelConfigFilepath)
		if os.path.exists(self.modelPklFilepath):
			os.remove(self.modelPklFilepath)

	# ************************* MAIN PRIVATE METHODS *************************

	def _train(self):
		"""
		Train/re-train self.model

		1. Load all the features and label from the database.
		2. Preprocess data by excluding rows with empty pay, taking the log 
			of pay to normalize the data, and excluding outliers.
		3. Train model with pipeline and save to the field, self.model.
		3. Save the field, self.model, to the file self.modelPklFilepath.
		4. Reset the dbChangeCounter and lastTrainedRowCount in the filepath
			self.modelConfigFilepath to 0.
		"""

		# ------------------ Load features and label from DB ------------------

		df = None
		try:
			with self.dbConn.cursor() as cursor:
				# sql query features and label
				cursor.execute("SELECT job_type, hours_worked, pay FROM jobs")
				df = pd.DataFrame(cursor.fetchall(), columns=["Job Type", "Hours Worked", "Pay"])
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in ML_MODEL._train():\n\n{e}", 
				terminate=True)
		
		# ------------------------ Data Preprocessing ------------------------

		# exclude rows where Pay is None (null) or 0
		df = df.dropna(subset=["Pay"])
		df = df[df["Pay"] != 0]

		# use log pay to normalize
		df["Log Pay"] = np.log(df["Pay"].astype(float))

		# exclude outliers
		Q1 = df["Log Pay"].quantile(0.25)
		Q3 = df["Log Pay"].quantile(0.75)
		IQR = Q3 - Q1
		lowerBound = Q1 - 1.5 * IQR
		upperBound = Q3 + 1.5 * IQR
		df = df[(df["Log Pay"] >= lowerBound) & (df["Log Pay"] <= upperBound)].copy()
		df = df.reset_index()

		# Seperate features and label
		features = df[["Job Type", "Hours Worked"]]
		label = df[["Log Pay"]]

		# --------------------------- Guard Clause ---------------------------

		# Do not train on empty dataset
		if len(df) == 0:
			return
		
		# --------------------- Train model with pipeline ---------------------

		# Preprocess the features columns (Job Type, Hours Worked)
		# One hot encoder for job type
		pipelineJobType = Pipeline([
			("Encoder", OneHotEncoder(handle_unknown="ignore")) 
		])
		# Standard scaler for hours worked
		pipelineHoursWorked = Pipeline([
			("Scaler", StandardScaler())
		])
		preprocessor = ColumnTransformer([
			("Formatted Job Type", pipelineJobType, ["Job Type"]),
			("Formatted Hours Worked", pipelineHoursWorked, ["Hours Worked"])
		])

		# Create ML model with preprocessor, assign to this class's self.model
		self.model = Pipeline([
			("Preprocessing", preprocessor),
			("Regressor", DecisionTreeRegressor(random_state=42))
		])
		
		# Train self.model on features and label
		self.model.fit(features, label)

		# -------------------- Save the model to model.pkl --------------------

		joblib.dump(self.model, self.modelPklFilepath)

		#  ---------------- Reset change counter and row count ----------------

		self._writeModelConfigFilepath({"dbChangeCounter": 0, 
								 "lastTrainedRowCount": len(label)})

	# ************************ PRIVATE HELPER METHODS ************************

	def _readModelConfigFilepath(self):
		"""
		Reads self.modelConfigFilepath and return the config (dict) if 
		available. Otherwise, create and return an empty one.

		Returns: config (dict).
		"""

		config = None

		if os.path.exists(self.modelConfigFilepath):
			with open(self.modelConfigFilepath, "r") as f:
				config = json.load(f)
		else:
			config = {"dbChangeCounter": 0, "lastTrainedRowCount": 0}
			with open(self.modelConfigFilepath, "w") as f:
				json.dump(config, f, indent=4)

		return config

	def _writeModelConfigFilepath(self, config):
		"""
		Writes the config (dict) to the json file in self.modelConfigFilepath.

		Args: 
			config (dict): Config object to be written to json file.
		"""

		with open(self.modelConfigFilepath, "w") as f:
			json.dump(config, f, indent=4)

	def _calculateRetrainLoadFactor(self):
		"""
		The Retrain Load Factor is a measure of how much entries in the 
		database has been altered since the last ML training. Simply, read the 
		config file and calculate.
		
		Returns:
			retrainLoadFactor (float): dbChangeCounter / lastTrainedRowCount.
		"""

		config = self._readModelConfigFilepath()
		dbChangeCounter = config["dbChangeCounter"]
		lastTrainedRowCount = config["lastTrainedRowCount"]
		
		if lastTrainedRowCount > 0:
			return dbChangeCounter / lastTrainedRowCount
		# if div by 0, then load factor is 0.0
		else:
			return 0.0

	def _predictAndCacheEverything(self):
		"""
		Reset and update the predicted pay in the cache for all IDs/entries. 
		This is used when something major happens such as, first time starting  
		the program or the ML model has been retrained.

		1. Retrieves the features of all IDs/entries
		2. Repredict using the ML model, self.model.
		3. Clear the cache and update the predicted pay in cache.
		"""

		# ------------- load every id and features from database -------------

		idList = None
		featuresList = None
		try:
			with self.dbConn.cursor() as cursor:
				cursor.execute("SELECT id, job_type, hours_worked FROM jobs")
				data = pd.DataFrame(cursor.fetchall(), columns=["ID", "Job Type", "Hours Worked"])
				idList = data["ID"].tolist()
				featuresList = data[["Job Type", "Hours Worked"]]
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in ML_MODEL._predictAndCacheEverything():\n\n{e}", 
				terminate=True)
			
		# --------------------------- Guard Clause ---------------------------

		# Do not predict an empty dataset
		if len(idList) == 0:
			return
		
		# ------------------------ predict everything ------------------------

		logPredictedPayList = self.model.predict(featuresList)
		predictedPayList = np.exp(logPredictedPayList)

		# ----------------------- cache the predictions -----------------------

		self.predictionCache.clear() # clear the cache first
		for i in range(len(idList)):
			id = idList[i]
			predictedPay = predictedPayList[i]
			self.predictionCache[id] = predictedPay

	def _predictAndCacheDirtyIDs(self, dirtyIDs):
		"""
		Updates the predicted pay in the cache for dirtyIDs.

		1. Retrieves the features of dirtyIDs.
		2. Repredict using the ML model, self.model.
		3. Update the predicted pay in cache.

		Args: 
			dirtyIDs (list): list of dirtyIDs to be repredicted.
		"""

		# --------------------------- Guard Clause ---------------------------

		# Do not predict on empty dataset
		if len(dirtyIDs) == 0:
			return

		# ---------------- Retrieve features of all dirtyIDs  ----------------

		idList = None
		featuresList = None
		try:
			with self.dbConn.cursor() as cursor:
				# Generate the sql query
				seperator = ","
				pieces = ["%s"] * len(dirtyIDs)
				placeholder = seperator.join(pieces) # example: "%s, %s, %s"
				sql = f"SELECT id, job_type, hours_worked FROM jobs WHERE id IN ({placeholder})"
				# Execute query and retrieve the data
				cursor.execute(sql, dirtyIDs)
				data = pd.DataFrame(cursor.fetchall(), columns=["ID", "Job Type", "Hours Worked"])
				idList = data["ID"].tolist()
				featuresList = data[["Job Type", "Hours Worked"]]
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in ML_MODEL._predictAndCacheDirtyIDs():\n\n{e}", 
				terminate=True)

		#  ------------------------- Predict dirtyIDs -------------------------

		logPredictedPayList = self.model.predict(featuresList)
		predictedPayList = np.exp(logPredictedPayList)

		# --------------------- Cache dirtyIDs prediction ---------------------

		for i in range(len(idList)):
			id = idList[i]
			predictedPay = predictedPayList[i]
			self.predictionCache[id] = predictedPay

class DateEditDelegate(QStyledItemDelegate):
	"""
	Delegate class following the QStyledItemDelegate template for dates in 
	QTableWidget's cell. 
	
	The purpose of using QDateEdit delegate on table cell 
	is to error-proof user input and cut overhead of rendering a QDateEdit. 
	This way, only if the cell is double clicked then a QDateEdit pops up.
	"""
	
	def createEditor(self, parent, option, index):
		"""
		Creates and returns QDateEdit that will be used when the user edits 
		a cell.

		Args:
			parent (QWidget): The parent widget (QTableWidget).
			option (QStyleOptionViewItem): Provides style options for the item 
											(not used).
			index (QModelIndex): The index of the cell being edited (not used).

		Returns: QDateEdit widget
		"""

		editor = QDateEdit(parent)
		editor.setCalendarPopup(True)
		editor.setDisplayFormat("MMM dd, yyyy")
		editor.setFocusPolicy(Qt.FocusPolicy.NoFocus)

		return editor

	def setEditorData(self, editor, index):
		"""
		Initializes the editor with the current cell's date value.

		Args:
			editor (QDateEdit): The editor widget created by createEditor.
			index (QModelIndex): The index of the cell being edited.
		"""

		dateStr = index.model().data(index)
		try:
			date = QDate.fromString(dateStr, "MMM dd, yyyy")
			editor.setDate(date)
		except:
			editor.setDate(QDate.currentDate())

	def setModelData(self, editor, model, index):
		"""
		Updates the model with the value chosen in the editor.

		Args:
			editor (QDateEdit): The editor widget used to edit the cell.
			model (QAbstractItemModel): The underlying model of the table.
			index (QModelIndex): The index of the cell being edited.
		"""

		model.setData(index, editor.date().toString("MMM dd, yyyy"))

class InsertDialog(QDialog):
	"""
	A class for pop up window when insert button is clicked on the database tab. 

	Provided fields for the user to enter input, error-checks the input, and 
	return the value of the input as a tuple.
	"""
	# ***************************** GUI and INITS *****************************

	def __init__(self):
		super().__init__()

		# ------------------------ Field Declarations ------------------------
		
		# UI fields
		self.layout = QGridLayout()
		self.dateInput = QDateEdit()
		self.jobNameInput = QLineEdit()
		self.jobTypeInput = QLineEdit()
		self.hoursWorkedInput = QLineEdit()
		self.payInput = QLineEdit()
		self.submitButton = QPushButton("Submit")
		self.cancelButton = QPushButton("Cancel")

		# --------------------------- Initialize UI ---------------------------
		
		self.initUI()

	def initUI(self):
		"""
		Initilize the format and how the UI of the Insert dialog looks.
		"""
		# Initialize QDialog fields
		self.setWindowTitle("Insert a Job Entry")
		self.setLayout(self.layout)

		# Initialize the date input
		self.dateInput.setCalendarPopup(True)
		self.dateInput.setDisplayFormat("MMM dd, yyyy")
		self.dateInput.setDate(QDate.currentDate())
		self.dateInput.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(QLabel("Date *"), 0, 0)
		self.layout.addWidget(self.dateInput, 0, 1)
		
		# Initialize the job name input
		self.layout.addWidget(QLabel("Job Name *"), 1, 0)
		self.layout.addWidget(self.jobNameInput, 1, 1)

		# Initialize the job type input
		self.layout.addWidget(QLabel("Job Type *"), 2, 0)
		self.layout.addWidget(self.jobTypeInput, 2, 1)

		# Initialize the hours Worked Input
		self.hoursWorkedInput.setValidator(QDoubleValidator(0.0, 24.0, 2))
		self.layout.addWidget(QLabel("Hours Worked *"), 3, 0)
		self.layout.addWidget(self.hoursWorkedInput, 3, 1)

		# Initialize the Pay Input
		self.payInput.setValidator(QDoubleValidator(0.0, 10000.0, 2))
		self.layout.addWidget(QLabel("Pay ($)"), 4, 0)
		self.layout.addWidget(self.payInput, 4, 1)

		# Initialize the submit button input
		self.submitButton.clicked.connect(self.handleSubmit)
		self.submitButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(self.submitButton, 5, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)

		# Initialize the cancel button input
		self.cancelButton.clicked.connect(self.confirmCancel)
		self.cancelButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(self.cancelButton, 6, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
	
	# **************************** EVENT HANDLERS ****************************

	def closeEvent(self, event):
		"""
		If the user try to close the window, like pressing the X button, call 
		on the cancel method to confirm cancellation. 
		Accept the QCloseEvent if the user wants to cancel, otherwise ignore 
		the QCloseEvent.

		Args:
			event (QCloseEvent): The close event passed by Qt when the user 
			try to close the window.
		"""
		if self.confirmCancel():
			event.accept() # accept the pressing X button event
		else: 
			event.ignore() # ignore the pressing X button event
	
	def confirmCancel(self):
		"""
		This method ask the user for confirmation to cancel entering the input.
		This is called when the cancel or the X button is pressed. Calls on 
		self.reject() to set its exec() False.
		
		Returns:
			True if user wants to cancel. False if user decided not to cancel.
		"""

		# if no entries entered, cancel right away. No need for question
		if (self.jobNameInput.text() == ""
				and self.jobTypeInput.text() == ""
				and self.hoursWorkedInput.text() == ""
				and self.payInput.text() == ""):
			self.reject() # to display False on exec()
			return True
		
		# Otherwise, double check with user to cancel entry
		reply = QMessageBox.question(
			self,
			"Confirm Close",
			"Are you sure you want to cancel this entry?",
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
		)

		if reply == QMessageBox.StandardButton.Yes:
			self.reject() # to display False on exec()
			return True # for cancel
		elif reply == QMessageBox.StandardButton.No:
			return False # for cancel

	def handleSubmit(self):
		"""
		Error-checks user input when the submit button is pressed. 
		If there are no errors, then self.accept() will be invoked to close the 
		dialog and signal True on exec().
		"""

		# ------------------- Guard clause: Input Checking -------------------

		# Job name cannot be empty
		if self.jobNameInput.text() == "":
			error(title="Edit Error", msg="Job name cannot be empty")
			return
		
		# Job type cannot be empty
		if self.jobTypeInput.text() == "":
			error(title="Edit Error", msg="Job type cannot be empty")
			return

		# Hours worked cannot be empty
		hoursWorkedInput = self.hoursWorkedInput.text()
		if hoursWorkedInput == "":
			error(title="Edit Error", msg="Hours worked cannot be empty")
			return
		
		# Checks Handled by QDoubleValidator: 
		# - Hours worked must be numeric
		# - Hours worked cannot be negative
		
		# Hours worked must be less than 24.0 hr
		if hoursWorkedInput == ".": # treat "." as "0" to be converted
			hoursWorkedInput = "0"
		if (float(hoursWorkedInput) > 24.0):
			error(title="Edit Error", msg="Hours worked must be between 0.00 to 24.00 hours")
			return
		
		# Checks Handled by QDoubleValidator: 
		# - Pay must be numeric
		# - Pay cannot be negative number if it is not empty
				
		# ---------------------------- Main Action ----------------------------

		# Signal that insert dialog closes successfully, good to retrieve data
		self.accept()

	# ******************************** GETTERS ********************************
	
	def getValuesAsTuple(self):
		"""
		Reads and returns user inputs as tuple, ready for database insertion.

		Returns:
			values (tuple): User inputs.
		"""

		date = self.dateInput.date().toString("yyyy-MM-dd")
		jobName = self.jobNameInput.text()
		jobType = self.jobTypeInput.text()
		hoursWorked = self.hoursWorkedInput.text()
		hoursWorked = 0 if hoursWorked == "." else float(hoursWorked)
		pay = self.payInput.text().replace("$", "").replace(" ", "").replace(",", "")
		if pay == ".": # treat "." as "0" to be converted
			pay = "0"
		pay = None if pay == "" else float(pay)

		return (date, jobName, jobType, hoursWorked, pay)
		
class DashboardTab(QWidget):
	"""
	This tab displays the ALL/YTD/year pay as a chart, and general statistics 
	including weekly pay, pay per job, and equivalent hourly rate.
	"""
	
	# ******************************** INITS ********************************

	def __init__(self, dbConn):
		super().__init__()	

		# ------------------------ Field Declarations ------------------------

		# Backend Fields
		self.dbConn = dbConn
		self.reRender = True # used for signaling if database has been changed

		# UI Fields
		self.layout = QGridLayout()
		# Chart Section
		self.chartSection = {
			"layout": QVBoxLayout(),
			"intervalComboBox": QComboBox(),
			"saveButton": QPushButton("Save Chart"),
			"totalLabel": QLabel(),
			"figure": Figure(figsize=(5,5)),
			"canvas": None,
			"ax": None
		}
		self.chartSection["canvas"] = FigureCanvas(self.chartSection["figure"])
		self.chartSection["ax"] = self.chartSection["figure"].add_subplot(111)

		# Pay Rate Section
		self.statsSection = {
			"mainWidget": QWidget(),
			"layout": QGridLayout(),
			"titleLabel": QLabel("Overall Statistics"),
			"spacer": QLabel(""),
			"avgWeeklyPayLabel": QLabel("Average Weekly Pay ($)"),
			"avgWeeklyPayLineEdit": QLineEdit(),
			"medianWeeklyPayLabel": QLabel("Median Weekly Pay ($)"),
			"medianWeeklyPayLineEdit": QLineEdit(),
			"avgPayPerJobLabel": QLabel("Average Pay Per Job ($)"),
			"avgPayPerJobLineEdit": QLineEdit(),
			"medianPayPerJobLabel": QLabel("Median Pay Per Job ($)"),
			"medianPayPerJobLineEdit": QLineEdit(),
			"equivalentPayRateLabel": QLabel("Equivalent Hourly Rate ($/hr)"),
			"equivalentPayRateLineEdit": QLineEdit()
		}

		# Init UI
		self.initUI()

		# Do initial render
		self.render()

	def initUI(self):
		styleSheet = """
			#chartTotalLabel {
				font-size: 20px;
			}

			#statsTitle {
				font-size: 20px;
				font-weight: bold;
			}

			#statsSection {
				background-color: palette(base);
				border-radius: 10px;
			}

			QLineEdit {
				background-color: palette(window);
			}
		"""

		self.setLayout(self.layout)

		# ------------------- Initialize the chart section -------------------

		# Initialize Interval Options (QComboBox)
		self.chartSection["intervalComboBox"].currentTextChanged.connect(self.renderChartSection)
		self.chartSection["intervalComboBox"].setFocusPolicy(Qt.FocusPolicy.NoFocus)
		# Initialize save button
		self.chartSection["saveButton"].clicked.connect(self.handleSaveToSvg)
		self.chartSection["saveButton"].setFocusPolicy(Qt.FocusPolicy.NoFocus)
		# Initialize the total label
		self.chartSection["totalLabel"].setObjectName("chartTotalLabel")
		self.chartSection["totalLabel"].setStyleSheet(styleSheet)
		self.chartSection["totalLabel"].setText(f"Total: $")
		# Configure the layout positioning
		self.chartSection["layout"].addWidget(self.chartSection["intervalComboBox"])
		self.chartSection["layout"].addWidget(self.chartSection["totalLabel"])
		self.chartSection["layout"].addWidget(self.chartSection["canvas"])
		self.chartSection["layout"].addWidget(self.chartSection["saveButton"])
		self.chartSection["layout"].addStretch(0)
		self.layout.addLayout(self.chartSection["layout"], 0, 0, 1, 1, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

		# ------------------ Initialize the pay rate section ------------------
		
		# Initialize the main widget/layout
		self.statsSection["mainWidget"].setStyleSheet(styleSheet)
		self.statsSection["mainWidget"].setObjectName("statsSection")
		self.statsSection["mainWidget"].setLayout(self.statsSection["layout"])
		# Initialize the title label
		self.statsSection["titleLabel"].setStyleSheet(styleSheet)
		self.statsSection["titleLabel"].setObjectName("statsTitle")
		# Initialize the weekly pay
		self.statsSection["avgWeeklyPayLineEdit"].setReadOnly(True)
		self.statsSection["medianWeeklyPayLineEdit"].setReadOnly(True)
		# Initialize pay per job
		self.statsSection["avgPayPerJobLineEdit"].setReadOnly(True)
		self.statsSection["medianPayPerJobLineEdit"].setReadOnly(True)
		# Initialize the hourly rates
		self.statsSection["equivalentPayRateLineEdit"].setReadOnly(True)
		# Configure the layout positioning
		self.statsSection["layout"].addWidget(self.statsSection["titleLabel"], 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
		self.statsSection["layout"].addWidget(self.statsSection["spacer"], 1, 0, 1, 2)
		self.statsSection["layout"].addWidget(self.statsSection["avgWeeklyPayLabel"], 2, 0, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["avgWeeklyPayLineEdit"], 2, 1, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["medianWeeklyPayLabel"], 3, 0, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["medianWeeklyPayLineEdit"], 3, 1, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["avgPayPerJobLabel"], 4, 0, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["avgPayPerJobLineEdit"], 4, 1, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["medianPayPerJobLabel"], 5, 0, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["medianPayPerJobLineEdit"], 5, 1, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["equivalentPayRateLabel"], 6, 0, 1, 1)
		self.statsSection["layout"].addWidget(self.statsSection["equivalentPayRateLineEdit"], 6, 1, 1, 1)
		self.layout.addWidget(self.statsSection["mainWidget"], 0, 1, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

	# ******************************* RENDERS *******************************

	def render(self):
		"""
		render orchestrate the rendering process of the dashboard tab.
		Starting with the chart section, then the stats section.
		"""
		self.renderChartSection()
		self.renderStatsSection()

		# Set this to False here because all the rendering is done
		self.reRender = False

	def renderChartSection(self):
		"""
		Renders the charts section including the dropdown options, label sum, 
		and the chart.

		Here are the steps that it takes:
		1. Read the dropdown for selected options (ALL, YTD, 2022, ....)
		2. Query the data based on the selected options
		3. Seperate and reformat the data into date and pay list
		4. Display the total as text
		5. Plot the chart (date vs cumulative pay)
		"""
		# -------------------- Read the selected options --------------------

		# If there has been database change or first time load up, 
		# render the chart options first.
		if self.reRender:
			self.renderChartIntervalOptions()

		# Get the current intervalselection of the comboBox
		intervalSelection = self.chartSection["intervalComboBox"].currentText()

		# ----------------- Retrieve data from the database -----------------

		rows = None
		try:
			with self.dbConn.cursor() as cursor:
				placeholder = ""
				args = []
				
				# Add options if applicable (YTD, ALL, 2022, ...)
				if intervalSelection == "YTD":
					placeholder = f" WHERE job_date >= MAKEDATE(YEAR(CURDATE()), 1) "
				elif intervalSelection == "ALL":
					placeholder = "" # No filter means all
				else:
					placeholder = f" WHERE year(job_date) = %s"
					args.append(intervalSelection)
				
				cursor.execute(f"SELECT job_date, pay FROM jobs {placeholder} ORDER BY job_date ASC", args)
				rows = cursor.fetchall()
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in DashboardTab.renderChartSection():\n\n{e}", 
				terminate=True)
		
		# ------------- Seperate and reformat dates and pay cols -------------

		dateList = []
		cumulativePayList = []
		paySum = 0.0
		for date, pay in rows:
			dateList.append(date)
			paySum += float(pay)
			cumulativePayList.append(round(paySum, 2))

		# ------------- Display the total pay (final cumulative) -------------

		totalText = "Total: $ "
		totalText += f"{cumulativePayList[-1]:,.2f}" if len(cumulativePayList) > 0 else ""
		self.chartSection["totalLabel"].setText(totalText)

		# -------------------------- Plot the chart --------------------------

		self.chartSection["ax"].clear()
		if (len(cumulativePayList) > 0):
			self.chartSection["ax"].plot(dateList, cumulativePayList)
			self.chartSection["ax"].set_title(f"Cumulative Pay ({intervalSelection})")
			# Configure date (x axis) based on all or yearly
			locator = mdates.AutoDateLocator() if intervalSelection == "ALL" else mdates.MonthLocator()
			self.chartSection["ax"].xaxis.set_major_locator(locator)
			self.chartSection["ax"].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
			self.chartSection["ax"].xaxis.set_visible(True)
			self.chartSection["figure"].autofmt_xdate()
			# Configure pay (y axis)
			self.chartSection["ax"].set_ylabel("Pay ($)")
			self.chartSection["figure"].tight_layout() # For the Y axis to display correctly
		else:
			self.chartSection["ax"].xaxis.set_visible(False)
		self.chartSection["canvas"].draw()

	def renderChartIntervalOptions(self):
		"""
		Renders interval options (All, YTD, 2022, ...) for the chart section.

		renderChartIntervalOptions is called when there is a database UPDATE, 
		INSERT, or DELETE to re evaluate the interval options.
		"""

		# Blocks the callback signal from the combo box for renderChartSection 
		# to avoid infinite loop.
		self.chartSection["intervalComboBox"].blockSignals(True)

		try:
			with self.dbConn.cursor() as cursor:
				# Populate Dropdown With Years Option: YTD, ALL, DATES
				cursor.execute("SELECT DISTINCT YEAR(job_date) AS year FROM jobs ORDER BY year DESC")
				years = [str(tup[0]) for tup in cursor.fetchall()]
				self.chartSection["intervalComboBox"].clear()
				self.chartSection["intervalComboBox"].addItems(["ALL", "YTD"])
				self.chartSection["intervalComboBox"].addItems(years)
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in DashboardTab.renderChartIntervalOptions():\n\n{e}", 
				terminate=True)
		
		self.chartSection["intervalComboBox"].blockSignals(False)
	
	def renderStatsSection(self):
		"""
		Render the values for the statistics section: 
		- weekly pay (average and median)
		- pay per job (average and median)
		- equivalent hourly rate
		"""

		try: 
			with self.dbConn.cursor() as cursor:
				# --------------- Guard Clause: return if empty ---------------

				cursor.execute("SELECT COUNT(*) FROM jobs")
				result = cursor.fetchone()
				count = result[0]

				if count == 0:
					self.statsSection["avgWeeklyPayLineEdit"].setText("")
					self.statsSection["medianWeeklyPayLineEdit"].setText("")
					self.statsSection["avgPayPerJobLineEdit"].setText("")
					self.statsSection["medianPayPerJobLineEdit"].setText("")
					self.statsSection["equivalentPayRateLineEdit"].setText("")
					return

				# ------------------- Render the weekly pay -------------------

				cursor.execute("SELECT SUM(pay) FROM jobs GROUP BY YEARWEEK(job_date)")
				data = [el[0] for el in cursor.fetchall()]
				self.statsSection["avgWeeklyPayLineEdit"].setText(f"{np.average(data):,.2f}")
				self.statsSection["medianWeeklyPayLineEdit"].setText(f"{np.median(data):,.2f}")

				# ------------------- Render the Pay per job ------------------

				cursor.execute("SELECT pay FROM jobs")
				data = [el[0] for el in cursor.fetchall()]
				self.statsSection["avgPayPerJobLineEdit"].setText(f"{np.average(data):,.2f}")
				self.statsSection["medianPayPerJobLineEdit"].setText(f"{np.median(data):,.2f}")

				# ------------- Render the equivalent hourly rate -------------

				cursor.execute("SELECT SUM(pay), SUM(hours_worked) FROM jobs")
				data = cursor.fetchone()
				hourlyRate = data[0] / data[1]
				self.statsSection["equivalentPayRateLineEdit"].setText(f"{hourlyRate:,.2f}")
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in DashboardTab.renderStatsSection():\n\n{e}", 
				terminate=True)
		
	# **************************** EVENT HANDLERS ****************************
	def handleSaveToSvg(self):
		"""
		Saves the current chart to SVG.
		"""

		# Ask user for filepath
		filepath, _ = QFileDialog.getSaveFileName(self,
						"Save To SVG",
						"",
						"SVG Files (*.svg);;All Files (*)"
					)
		
		# return if canceled 
		if (filepath == ""):
			return

		# Main action: save to svg 
		self.chartSection["figure"].savefig(filepath)

class DatabaseTab(QWidget):
	"""
	GUI representation of database in QTableWidget along with interactivity 
	with SQL queries such as: SELECT, UPDATE, INSERT, DELETE.

	SELECT - user can filter, sort, and/or search entries in the database.
	UPDATE - user can double click on table cell to edit entry.
	INSERT - user can insert new entry into the database.
	DELETE - user can delete entry/entries from the database.
	"""

	# ***************************** GUI and INITS *****************************

	def __init__(self, dbConn, MLmodel, dashboardTab):
		super().__init__()
		
		# ------------------------ Field Declarations ------------------------

		# Backend Fields
		self.dbConn = dbConn # database sql connection
		self.MLmodel = MLmodel
		self.dashboardTab = dashboardTab # ref to the dashboard to signal db change
		
		# UI Fields
		# General UI Fields
		self.layout = QGridLayout()
		self.searchBox = QLineEdit()
		self.deleteButton = QPushButton("Delete")
		self.insertButton = QPushButton("Insert")
		self.saveToCsvButton = QPushButton("Save to CSV")
		self.table = QTableWidget()
		# Filters UI
		self.filterScrollArea = QScrollArea()
		self.filterLabel = QLabel("Filter By")
		self.filterByDate = {
			"checkBox": QCheckBox("Date"),
			"fromLabel": QLabel("From"),
			"toLabel": QLabel("To"),
			"fromDateEdit": QDateEdit(),
			"toDateEdit": QDateEdit(),
			"spacer": QLabel("")
		}
		self.filterByJobType = {
			"checkBox": QCheckBox("Job Type"),
			"comboBox": QComboBox(),
			"spacer": QLabel("")
		}
		self.filterByHoursWorked = {
			"checkBox": QCheckBox("Hours Worked"),
			"fromLabel": QLabel("From"),
			"toLabel": QLabel("To"),
			"fromLineEdit": QLineEdit(),
			"toLineEdit": QLineEdit(),
			"spacer": QLabel("")
		}
		self.filterByPay = {
			"checkBox": QCheckBox("Pay ($)"),
			"fromLabel": QLabel("From"),
			"toLabel": QLabel("To"),
			"fromLineEdit": QLineEdit(),
			"toLineEdit": QLineEdit(),
			"spacer": QLabel("")
		}
		# Sorts UI
		self.sortButtonsGroup = QButtonGroup()
		self.sortButtons = { # Note: defined in a dict so it can be looped
				"Date Newest to Oldest": QRadioButton("Date Newest to Oldest"),
				"Date Oldest to Newest": QRadioButton("Date Oldest to Newest"),
				"Job Name A-Z": QRadioButton("Job Name A-Z"),
				"Job Name Z-A": QRadioButton("Job Name Z-A"),
				"Job Type A-Z": QRadioButton("Job Type A-Z"),
				"Job Type Z-A": QRadioButton("Job Type Z-A"),
				"Hours Worked Lowest to Highest": QRadioButton("Hours Worked Lowest to Highest"),
				"Hours Worked Highest to Lowest": QRadioButton("Hours Worked Highest to Lowest"),
				"Pay Lowest to Highest": QRadioButton("Pay Lowest to Highest"),
				"Pay Highest to Lowest": QRadioButton("Pay Highest to Lowest")
			}

		# --------------------------- Initialize UI ---------------------------

		self.initGeneralUI() # Table, search bar, insert delete save buttons
		self.initFiltersUI() # "Filter by" section
		self.initSortsUI() # "Sort by" section

		# Render after initialized
		self.renderFiltersUI() # renders checkboxes and its contents if checked
		
	def initGeneralUI(self):
		"""
		Initializes search bar, delete buttom, insert button, Save to CSV 
		button, and table for formatting and connects.
		"""

		# CSS for this page
		styleSheet = """
			QTableWidget {
				 min-width: 800px
			}
			"""

		# Initialize layout
		self.setLayout(self.layout)

		# Initialize job search field
		self.searchBox.setPlaceholderText("Search Job Name")
		self.searchBox.textChanged.connect(self.renderTable)
		self.layout.addWidget(self.searchBox, 0, 0, 1, 4)

		# Initialize delete button
		self.deleteButton.clicked.connect(self.sqlDeleteActions)
		self.deleteButton.setEnabled(False)
		self.deleteButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(self.deleteButton, 1, 1, alignment=Qt.AlignmentFlag.AlignHCenter)

		# Initialize insert button
		self.insertButton.clicked.connect(self.sqlInsertActions)
		self.insertButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(self.insertButton, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter)

		# Initialize save to csv button
		self.saveToCsvButton.clicked.connect(self.handleSaveToCsv)
		self.saveToCsvButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(self.saveToCsvButton, 1, 3, alignment=Qt.AlignmentFlag.AlignHCenter)

		# Initialize the table
		self.table.setColumnCount(6)
		self.table.setHorizontalHeaderLabels(["Date", "Job Name", "Job Type", "Hours Worked", "Pay", "Predicted Pay"])
		self.table.setStyleSheet(styleSheet)
		self.table.itemChanged.connect(self.sqlUpdateActions)
		self.table.selectionModel().selectionChanged.connect(self.toggleDeleteButtonClickability)
		self.layout.addWidget(self.table, 2, 1, 9, 3)

	def initFiltersUI(self):
		"""
		Initializes the "Filter By" components like its date format, 
		allowed inputs, connects, and styling.
		"""

		styleSheet = """
			QScrollArea {
				min-width: 350px;
				max-width: 350px;
			}

			QLabel#filterLabel {
				font-size: 18px;
				font-weight: bold;
			}

			QCheckBox[class="sublabel"] {
				font-weight: bold;
			}
			"""
		
		# Initialize scroll area format and its parent
		self.filterScrollArea.setWidgetResizable(False)
		self.filterScrollArea.setStyleSheet(styleSheet)
		self.layout.addWidget(self.filterScrollArea, 1, 0, 6, 1)
		
		# Initialize Filter label format
		self.filterLabel.setObjectName("filterLabel")
		self.filterLabel.setStyleSheet(styleSheet)

		# Initialize date filter format and connects
		self.filterByDate["checkBox"].setProperty("class", "sublabel")
		self.filterByDate["checkBox"].setStyleSheet(styleSheet)
		self.filterByDate["checkBox"].toggled.connect(lambda: self.renderFiltersUI(self.filterByDate))
		self.filterByDate["fromDateEdit"].setCalendarPopup(True)
		self.filterByDate["fromDateEdit"].setDisplayFormat("MMM dd, yyyy")
		self.filterByDate["fromDateEdit"].dateChanged.connect(self.renderTable)
		self.filterByDate["fromDateEdit"].setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.filterByDate["toDateEdit"].setCalendarPopup(True)
		self.filterByDate["toDateEdit"].setDisplayFormat("MMM dd, yyyy")
		self.filterByDate["toDateEdit"].dateChanged.connect(self.renderTable)
		self.filterByDate["toDateEdit"].setFocusPolicy(Qt.FocusPolicy.NoFocus)

		# Initialize Job Type filter format and connects
		self.filterByJobType["checkBox"].setProperty("class", "sublabel")
		self.filterByJobType["checkBox"].setStyleSheet(styleSheet)
		self.filterByJobType["checkBox"].toggled.connect(lambda: self.renderFiltersUI(self.filterByJobType))
		self.filterByJobType["comboBox"].currentIndexChanged.connect(self.renderTable)

		# Initialize Hours Worked filter format and connects
		self.filterByHoursWorked["checkBox"].setProperty("class", "sublabel")
		self.filterByHoursWorked["checkBox"].setStyleSheet(styleSheet)
		self.filterByHoursWorked["checkBox"].toggled.connect(lambda: self.renderFiltersUI(self.filterByHoursWorked))
		self.filterByHoursWorked["fromLineEdit"].setValidator(QDoubleValidator(0.0, 24.0, 2))
		self.filterByHoursWorked["fromLineEdit"].textChanged.connect(self.renderTable)
		self.filterByHoursWorked["toLineEdit"].setValidator(QDoubleValidator(0.0, 24.0, 2))
		self.filterByHoursWorked["toLineEdit"].textChanged.connect(self.renderTable)

		# Initialize Pay filter format and connects
		self.filterByPay["checkBox"].setProperty("class", "sublabel")
		self.filterByPay["checkBox"].setStyleSheet(styleSheet)
		self.filterByPay["checkBox"].toggled.connect(lambda: self.renderFiltersUI(self.filterByPay))
		self.filterByPay["fromLineEdit"].setValidator(QDoubleValidator(0.0, 10000.0, 2))
		self.filterByPay["fromLineEdit"].textChanged.connect(self.renderTable)
		self.filterByPay["toLineEdit"].setValidator(QDoubleValidator(0.0, 10000.0, 2))
		self.filterByPay["toLineEdit"].textChanged.connect(self.renderTable)

	def initSortsUI(self):
		"""
		Initializes the sort by components for formatting and connects. 
		"""
		styleSheet = """
			QScrollArea {
				min-width: 350px;
				max-width: 350px;
			}

			QLabel#sortLabel {
				font-size: 18px;
				font-weight: bold;
			}

			QRadioButton[class="sublabel"] {
				font-weight: bold;
			}
			"""
		
		sortLayout = QGridLayout()
		
		# Initialize Sort label
		sortLabel = QLabel("Sort By")
		sortLabel.setObjectName("sortLabel")
		sortLabel.setStyleSheet(styleSheet)
		sortLayout.addWidget(sortLabel, 0, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)

		# Initialize all sort buttons
		self.sortButtons["Date Newest to Oldest"].setChecked(True) # check this one initially
		for i, button in enumerate(self.sortButtons.values()):
			button.setProperty("class", "sublabel")
			button.setStyleSheet(styleSheet)
			sortLayout.addWidget(button, i + 1, 0, 1, 2, alignment=Qt.AlignmentFlag.AlignLeft)
			self.sortButtonsGroup.addButton(button)
		self.sortButtonsGroup.buttonClicked.connect(self.renderTable)

		# Initialize Layout hierarchy (-> is into)
		# sortLayout layout -> widget -> scrollArea -> main layout
		sortWidget = QWidget()
		sortWidget.setLayout(sortLayout)
		sortScrollArea = QScrollArea()
		sortScrollArea.setWidget(sortWidget)
		sortScrollArea.setWidgetResizable(False)
		sortScrollArea.setStyleSheet(styleSheet)
		self.layout.addWidget(sortScrollArea, 7, 0, 4, 1)

	def renderFiltersUI(self, currentDataUpdate = None):
		"""
		Render the UI for selected "filter by" items. 
		For example, if filter by date is selected, render the QDateEdits for 
		the user to select date range. Hide everything else.
		
		Note: This is seperated from initFilterUI because this has to be called 
		multiple times to render what is being checked, whereas initFiltersUI 
		is only used to initialize the components.

		Args:
			currentDataUpdate (dict) - reference to the dict sections that is 
				rendered for the first time. This is important to clear off 
				previous filters. Valid dict sections are self.filterByDate, 
				self.filterByJobType, self.filterByHoursWorked, self.filterByPay
				
		"""

		# --------------- Create a new filter widget and layout ---------------

		# In the tree hierarchy, old widget->layout->childs will be deallocated
		newFilterWidget = QWidget()
		newFilterLayout = QGridLayout()

		# ----------------------- Add "Filter By" label -----------------------

		newFilterLayout.addWidget(self.filterLabel, 0, 0, 1, 2)
		# newFilterLayout.addWidget(QLabel(""), 1, 0, 1, 2) # spacer
		
		# -------------------- Add Filter By Date layouts --------------------

		newFilterLayout.addWidget(self.filterByDate["checkBox"], 2, 0, 1, 2)

		if self.filterByDate["checkBox"].isChecked():
			# Update the from and to QDates from oldest to newest if marked by currentDataUpdate
			if currentDataUpdate is self.filterByDate: # compare references
				oldestDate = None
				newestDate = None
				try:
					with self.dbConn.cursor() as cursor:
						cursor.execute("SELECT MIN(job_date) FROM jobs")
						rows = cursor.fetchone()
						oldestDate = rows[0]
						cursor.execute("SELECT MAX(job_date) FROM jobs")
						rows = cursor.fetchone()
						newestDate = rows[0]
				except Exception as e:
					error(title="Database Error", 
						msg=f"Database Error in DatabaseTab.renderFiltersUI():\n\n{e}", 
						terminate=True)

				self.filterByDate["fromDateEdit"].setDate(QDate(oldestDate.year, oldestDate.month, oldestDate.day))
				self.filterByDate["toDateEdit"].setDate(QDate(newestDate.year, newestDate.month, newestDate.day))

			# Append filter by date components to the new layout
			newFilterLayout.addWidget(self.filterByDate["fromLabel"], 3, 0, 1, 1)
			newFilterLayout.addWidget(self.filterByDate["toLabel"], 3, 1, 1, 1)
			newFilterLayout.addWidget(self.filterByDate["fromDateEdit"], 4, 0, 1, 1)
			newFilterLayout.addWidget(self.filterByDate["toDateEdit"], 4, 1, 1, 1)
			newFilterLayout.addWidget(self.filterByDate["spacer"], 5, 0, 1, 2)
		else:
			# Set childs' parent to None to prevent deallocation
			self.filterByDate["fromLabel"].setParent(None)
			self.filterByDate["toLabel"].setParent(None)
			self.filterByDate["fromDateEdit"].setParent(None)
			self.filterByDate["toDateEdit"].setParent(None)
			self.filterByDate["spacer"].setParent(None)

		# ------------------ Add Filter By Job Type layouts ------------------

		newFilterLayout.addWidget(self.filterByJobType["checkBox"], 6, 0, 1, 1)

		if self.filterByJobType["checkBox"].isChecked():
			# Initialize the QComboBox with all possible job types from db if marked as currentDataUpdate
			if currentDataUpdate is self.filterByJobType:
				jobTypeList = None
				try:
					with self.dbConn.cursor() as cursor:
						cursor.execute("SELECT DISTINCT job_type FROM jobs")
						rows = cursor.fetchall()
						jobTypeList = [col[0] for col in rows]
				except Exception as e:
					error(title="Database Error", 
						msg=f"Database Error in DatabaseTab.renderFiltersUI():\n\n{e}", 
						terminate=True)
				
				self.filterByJobType["comboBox"].clear()
				self.filterByJobType["comboBox"].addItem("All")
				for jobType in jobTypeList:
					self.filterByJobType["comboBox"].addItem(jobType)

			# Append filter by job type components to the new layout
			newFilterLayout.addWidget(self.filterByJobType["comboBox"], 6, 1, 1, 1)
			newFilterLayout.addWidget(self.filterByJobType["spacer"], 8, 0, 1, 2)
		else:
			# Set childs' parent to None to prevent deallocation
			self.filterByJobType["comboBox"].setParent(None)
			self.filterByJobType["spacer"].setParent(None)

		# ---------------- Add Filter By Hours Worked layouts ----------------

		newFilterLayout.addWidget(self.filterByHoursWorked["checkBox"], 9, 0, 1, 2)

		if self.filterByHoursWorked["checkBox"].isChecked():
			# clears out old filter data if marked for currentDataUpdate
			if currentDataUpdate is self.filterByHoursWorked:
				self.filterByHoursWorked["fromLineEdit"].setText("")
				self.filterByHoursWorked["toLineEdit"].setText("")
			
			# append widgets to new layout
			newFilterLayout.addWidget(self.filterByHoursWorked["fromLabel"], 10, 0, 1, 1)
			newFilterLayout.addWidget(self.filterByHoursWorked["toLabel"], 10, 1, 1, 1)
			newFilterLayout.addWidget(self.filterByHoursWorked["fromLineEdit"], 11, 0, 1, 1)
			newFilterLayout.addWidget(self.filterByHoursWorked["toLineEdit"], 11, 1, 1, 1)
			newFilterLayout.addWidget(self.filterByHoursWorked["spacer"], 12, 0, 1, 2)
		else:
			# Set childs' parent to None to prevent deallocation
			self.filterByHoursWorked["fromLabel"].setParent(None)
			self.filterByHoursWorked["toLabel"].setParent(None)
			self.filterByHoursWorked["fromLineEdit"].setParent(None)
			self.filterByHoursWorked["toLineEdit"].setParent(None)
			self.filterByHoursWorked["spacer"].setParent(None)

		# ---------------- Add Filter By Hours Worked layouts ----------------

		newFilterLayout.addWidget(self.filterByPay["checkBox"], 13, 0, 1, 2)

		if self.filterByPay["checkBox"].isChecked():
			# clears out old filter data if marked for currentDataUpdate
			if currentDataUpdate is self.filterByPay:
				self.filterByPay["fromLineEdit"].setText("")
				self.filterByPay["toLineEdit"].setText("")
			
			# append widgets to new layout
			newFilterLayout.addWidget(self.filterByPay["fromLabel"], 14, 0, 1, 1)
			newFilterLayout.addWidget(self.filterByPay["toLabel"], 14, 1, 1, 1)
			newFilterLayout.addWidget(self.filterByPay["fromLineEdit"], 15, 0, 1, 1)
			newFilterLayout.addWidget(self.filterByPay["toLineEdit"], 15, 1, 1, 1)
			newFilterLayout.addWidget(self.filterByPay["spacer"], 16, 0, 1, 2)
		else:
			# Set childs' parent to None to prevent deallocation
			self.filterByPay["fromLabel"].setParent(None)
			self.filterByPay["toLabel"].setParent(None)
			self.filterByPay["fromLineEdit"].setParent(None)
			self.filterByPay["toLineEdit"].setParent(None)
			self.filterByPay["spacer"].setParent(None)

		# ------------- Deallocate old filter widget and set new -------------

		newFilterWidget.setLayout(newFilterLayout)
		self.filterScrollArea.setWidget(newFilterWidget) # this will deallocate old filter widget

		# Render table to show changes
		self.renderTable()

	def renderTable(self):
		"""
		Render the table with filtered and sorted (if any) database entries.
		
		Here are the steps it takes:
		1. It calls on sqlSelectActions to retrieve data based on filters and 
			sorts.
		2. Then, it calls on MLModel.predict for the predicted pays column.
		3. Display the data on the table.
		"""

		# block signal to not trigger itemChanged
		self.table.blockSignals(True)

		# fetch the entries based on filters and sorts from db
		data = self.sqlSelectActions()

		# Clears current table and allocate new rows for new data
		self.table.setRowCount(0) 
		self.table.setRowCount(len(data))

		# Ask ML for predictions
		ids = [tup[0] for tup in data] # 0th index of each tuple is id
		predictedPayList = self.MLmodel.predict(ids)
		
		# Display cells on table
		for i in range(len(data)):
			# index 0: id (DO NOT DISPLAY!!!)
			id = data[i][0]

			# index 1: job date
			jobDate = data[i][1] # type of datetime.date
			# Reformat to something like "Jun 01, 2025" in string format
			jobDateFormattedStr = jobDate.strftime("%b %d, %Y")
			# Set the cell on table
			dateItem = QTableWidgetItem(jobDateFormattedStr)
			self.table.setItem(i, 0, dateItem)
			# Save id into date col (col 0) only to save memory
			dateItem.setData(Qt.ItemDataRole.UserRole, id)

			# index 2: job name
			jobName = data[i][2]
			self.table.setItem(i, 1, QTableWidgetItem(jobName))

			# index 3: job type
			jobType = data[i][3]
			self.table.setItem(i, 2, QTableWidgetItem(jobType))

			# index 4: hours worked
			hoursWorked = data[i][4]
			self.table.setItem(i, 3, QTableWidgetItem(f"{hoursWorked:.2f}"))

			# index 5: pay
			pay = data[i][5]
			if pay != None: 
				self.table.setItem(i, 4, QTableWidgetItem(f"$ {pay:,.2f}"))

			# From ML prediction: predicted pay
			predictedPay = predictedPayList[id]
			item = QTableWidgetItem(f"$ {predictedPay:,.2f}")
			# make ML prediction column not editable
			item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
			self.table.setItem(i, 5, item)
		
		# Create delegate for the job date column
		delegate = DateEditDelegate(self.table)
		self.table.setItemDelegateForColumn(0, delegate)

		# Resize columns
		self.table.resizeColumnsToContents() # Fit to content for the rest
		self.table.setColumnWidth(0, 130) # Fixed size for date column

		# unblock signal to allow itemChanged
		self.table.blockSignals(False) 
	
	def restoreOldCellValue(self, row_i, col_i, oldValue):
		"""
		Restore the value of cell at row_i col_i to oldValue.

		Args: 
			row_i (int): Row index of the cell.
			col_i (int): Column index of the cell.
			oldValue (str): The previous value as a string.
		"""

		# use blockSignals to prevent repeated callbacks when cell is changed
		self.table.blockSignals(True)
		self.table.setItem(row_i, col_i, QTableWidgetItem(oldValue))
		self.table.blockSignals(False)

	# ****************************** SQL ACTIONS ******************************

	def sqlSelectActions(self):
		"""
		Perform SQL SELECT query to the database with optional filters and sort.

		Here are the steps it takes:
		1. Read the "Filter By" components and compile a list of filters
		2. Read the "Sort By" component keep track of what is selected.
		3. Create sql query based on filter and sort.
		4. Return matched rows.

		Returns:
			rows (list of tuple) - retrieved database entries matching on 
				filters and sorts.
		"""
		try:
			with self.dbConn.cursor() as cursor:
				# ------------------------- Filters -------------------------

				filterList = [] 
				filterArgList = []

				searchText = self.searchBox.text()
				if searchText != "":
					filterList.append("LOWER(job_name) LIKE %s")
					filterArgList.append(f"%{searchText.lower()}%")
				
				if self.filterByDate["checkBox"].isChecked():
					fromDate = self.filterByDate["fromDateEdit"].date().toString("yyyy-MM-dd")
					toDate = self.filterByDate["toDateEdit"].date().toString("yyyy-MM-dd")
					filterList.append("job_date BETWEEN %s AND %s")
					filterArgList.append(fromDate)
					filterArgList.append(toDate)
				
				if self.filterByJobType["checkBox"].isChecked():
					selection = self.filterByJobType["comboBox"].currentText()
					if (selection != "All"):
						filterList.append("job_type = %s")
						filterArgList.append(selection)

				if self.filterByHoursWorked["checkBox"].isChecked():
					fromStr = self.filterByHoursWorked["fromLineEdit"].text()
					if fromStr != "" and fromStr != ".":
						filterList.append("hours_worked >= %s")
						filterArgList.append(float(fromStr))

					toStr = self.filterByHoursWorked["toLineEdit"].text()
					if toStr != "" and toStr != ".":
						filterList.append("hours_worked <= %s")
						filterArgList.append(float(toStr))

				if self.filterByPay["checkBox"].isChecked():
					fromStr = self.filterByPay["fromLineEdit"].text()
					if fromStr != "" and fromStr != ".":
						filterList.append("pay >= %s")
						filterArgList.append(float(fromStr))

					toStr = self.filterByPay["toLineEdit"].text()
					if toStr != "" and toStr != ".":
						filterList.append("pay <= %s")
						filterArgList.append(float(toStr))

				# --------------------------- Sorts ---------------------------

				sortKey = self.sortButtonsGroup.checkedButton().text()
				sortMap = {
					"Date Newest to Oldest": "job_date DESC",
					"Date Oldest to Newest": "job_date ASC",
					"Job Name A-Z": "job_name ASC",
					"Job Name Z-A": "job_name DESC",
					"Job Type A-Z": "job_type ASC",
					"Job Type Z-A": "job_type DESC",
					"Hours Worked Lowest to Highest": "hours_worked ASC",
					"Hours Worked Highest to Lowest": "hours_worked DESC",
					"Pay Lowest to Highest": "pay ASC",
					"Pay Highest to Lowest": "pay DESC"
				}
				sortSql = " ORDER BY " + sortMap[sortKey]

				# ----------------------- SELECT Query -----------------------

				sql = "SELECT * FROM jobs"
				if len(filterArgList):
					sql += " WHERE "
					sql += " AND ".join(filterList)
				sql += sortSql

				cursor.execute(sql, filterArgList)
				rows = cursor.fetchall()
				return rows
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in DatabaseTab.sqlSelectActions():\n\n{e}", 
				terminate=True)

	def sqlUpdateActions(self, item):
		"""
		Perform SQL UPDATE query to edit the cell in the database.

		Here are the actions that will be triggered:
		1. Error Checking: First, it checks if the data type of item is 
			compatible in the database. For example, item must be numeric, 
			cannot be null. Error out if incompatible.
		2. SQL: UPDATE item into the database.
		3. ML: Increment ML changes counter if feature or label is changed. 
			Additionally, call ML predict to update cache if feature is changed.
		4. Dashboard Tab: Signal that database has been changed to rerender 
			the next time user visits dashboard.
		5. GUI: Re-render the table for sorts, filter, and new ML prediction.

		Args: 
			item (QTableWidgetItem): Object passed by PyQt containing 
				informations such as row index, col index, text, etc., 
				which would be error-checked before UPDATED into the database.
		"""

		#  ------------ Get relevant variables for error checkings ------------

		columnHeaderDB = ["job_date", "job_name", "job_type", "hours_worked", "pay"]
		row_i = item.row()
		col_i = item.column()
		# Note: the id is only stored on column 0 (date) to save memory
		id = self.table.item(row_i, 0).data(Qt.ItemDataRole.UserRole)
		newValue = item.text().strip() # remove leading/trailing whitespace
		# Retrieve old value from the database in case we need to restore
		oldValue = ""
		try:
			with self.dbConn.cursor() as cursor:
				sql = f"SELECT {columnHeaderDB[col_i]} FROM jobs WHERE id = %s"
				cursor.execute(sql, (id,))
				results = cursor.fetchall() # a list of tuple
				oldValue = str(results[0][0]) # 1st item in 1st tuple in 1st list
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in DatabaseTab.sqlUpdateActions():\n\n{e}", 
				terminate=True)
		
		# ------------------------- Error Checkings  -------------------------

		if columnHeaderDB[col_i] == "job_date":
			# No need to error check since picked by date picker (error-proof)
			# Just re-format
			newValue = datetime.strptime(newValue, "%b %d, %Y").strftime("%Y-%m-%d")
		elif columnHeaderDB[col_i] == "job_name":
			# Job name cannot be empty
			if newValue == "":
				self.restoreOldCellValue(row_i, col_i, oldValue)
				error(title="Edit Error", msg="Job name cannot be empty")
				return
		elif columnHeaderDB[col_i] == "job_type":
			# Job type cannot be empty
			if newValue == "":
				self.restoreOldCellValue(row_i, col_i, oldValue)
				error(title="Edit Error", msg="Job type cannot be empty")
				return
		elif columnHeaderDB[col_i] == "hours_worked":
			# Hours worked cannot be empty
			if newValue == "":
				self.restoreOldCellValue(row_i, col_i, oldValue)
				error(title="Edit Error", msg="Hours worked cannot be empty")
				return
			# Hours worked must be Numeric
			try:
				if newValue == ".": # treat "." as "0" to be converted
					newValue = "0"
				newValue = float(newValue)
			except ValueError:
				self.restoreOldCellValue(row_i, col_i, oldValue)
				error(title="Edit Error", msg="Hours worked must be numeric")
				return
			# Hours worked must be between 0.0 to 24.0 hr
			if (newValue < 0.0 or newValue > 24.0):
				self.restoreOldCellValue(row_i, col_i, oldValue)
				error(title="Edit Error", msg="Hours worked must be between 0.00 to 24.00 hours")
				return
		elif columnHeaderDB[col_i] == "pay":
			# Remove "$", " ", "," from newValue because of formatting
			newValue = newValue.replace("$", "").replace(" ", "").replace(",", "")
			# Pay must be positive numeric OR empty
			if newValue == "":
				newValue = None
			else:
				# Pay must be numeric
				try:
					if newValue == ".": # treat "." as "0" to be converted
						newValue = "0"
					newValue = float(newValue)
				except ValueError:
					oldValue = f"$ {float(oldValue):,.2f}" if oldValue != "None" else ""
					self.restoreOldCellValue(row_i, col_i, oldValue)
					error(title="Edit Error", msg="Pay must be numeric")
					return
				# Pay cannot be negative
				if newValue < 0.0:
					oldValue = f"$ {float(oldValue):,.2f}" if oldValue != "None" else ""
					self.restoreOldCellValue(row_i, col_i, oldValue)
					error(title="Edit Error", msg="Pay cannot be negative number")
					return

		# ----------------------------- DB UPDATE -----------------------------

		try: 
			with self.dbConn.cursor() as cursor:
				sql = f"UPDATE jobs SET {columnHeaderDB[col_i]} = %s WHERE id = %s"
				cursor.execute(sql, (newValue, id))
				self.dbConn.commit()
		except Exception as e:
			self.dbConn.rollback()
			error(title="Database Error", 
				msg=f"Database Error in DatabaseTab.sqlUpdateActions():\n\n{e}", 
				terminate=True)
		
		# -------------------- ML increment and re-predict --------------------

		# if the edited column is feature or label, increment changes counter
		if (columnHeaderDB[col_i] == "job_type" 
	  			or columnHeaderDB[col_i] == "hours_worked"
				or columnHeaderDB[col_i] == "pay"):
			self.MLmodel.incrementDbChangeCounter()
		
		# If the edited column is a feature, mark it dirty so it will be re-predicted
		if (columnHeaderDB[col_i] == "job_type" 
	  			or columnHeaderDB[col_i] == "hours_worked"):
			self.MLmodel.markDirtyIDs([id])
		
		# ----------------- Signal DB Change to Dashboard Tab -----------------

		self.dashboardTab.reRender = True

		# -------------- Re-renders table for sorts and filters --------------

		# callback to refresh table after all UI call stack is complete.
		# (similar to setTimeOut in javascript)
		QTimer.singleShot(0, self.renderTable)

	def sqlInsertActions(self):
		"""
		Perform SQL INSERT query to insert a new entry to the database.
		
		Here are the steps it takes:
		1. Create an InsertDialog object and call it (exec()).
		2. Let user enter the inputs and wait. User input is error checked by 
			InsertDialog class.
		3. Once successfully submitted, retrieve user input as tuple.
		4. Insert tuple into the database.
		5. Increment ML change counter.
		6. Dashboard Tab: Signal that database has been changed to rerender 
			the next time user visits dashboard.
		7. Re-render table to show effect.
		"""
		dialog = InsertDialog()

		# --------------------------- Guard Clause ---------------------------

		# if dialog status rejected (closed or cancel button), then cancel
		if dialog.exec() == False:
			return
		
		# ----------------------------- Main Code -----------------------------
		
		# Insert into database
		try: 
			with self.dbConn.cursor() as cursor:
				sql = """
					INSERT INTO jobs (job_date, job_name, job_type, hours_worked, pay)
					VALUES (%s, %s, %s, %s, %s)
				"""
				cursor.execute(sql, dialog.getValuesAsTuple())
				self.dbConn.commit()
		except Exception as e:
			self.dbConn.rollback()
			error(title="Database Error", 
				msg=f"Database Error in DatabaseTab.sqlInsertActions():\n\n{e}", 
				terminate=True)
			
		# Increment ML change counter
		self.MLmodel.incrementDbChangeCounter()

		# Signal DB Change to Dashboard Tab
		self.dashboardTab.reRender = True

		# Re render table to display inserted entry
		self.renderTable()

	def sqlDeleteActions(self):
		"""
		Perform SQL DELETE query to delete entry/entries from the database.

		Here are the steps that it takes:
		1. Get the rows indexes that are selected.
		2. Confirm with the user for deletion.
		3. Delete entry on database.
		4. Increment ML db change counter.
		5. Dashboard Tab: Signal that database has been changed to rerender 
			the next time user visits dashboard.
		6. Re-render table to show effect.
		"""
		# ------------------------ Get Selected Items ------------------------

		# Retrieve the selected row indexes, IDs, and job names (python style)
		selectedRowIndexes = np.unique([index.row() for index in self.table.selectedIndexes()])
		selectedIDs = [self.table.item(i, 0).data(Qt.ItemDataRole.UserRole) for i in selectedRowIndexes]
		selectedJobNames = [self.table.item(i, 1).text() for i in selectedRowIndexes]

		# --------------- Display Delete Confirmation Question ---------------

		title = "Confirm Delete Entry" if len(selectedIDs) == 1 else "Confirm Delete Entries"

		text = "Are you sure you want to delete "
		if (len(selectedIDs) == 1):
			text += f'"{selectedJobNames[0]}" ?'
		elif (len(selectedIDs) == 2):
			text += f'"{selectedJobNames[0]}" and "{selectedJobNames[1]}" ?'
		else:
			text += f'"{selectedJobNames[0]}" ... "{selectedJobNames[-1]}" ({len(selectedJobNames)} entries total) ?'
		text += "\n\nThis action cannot be undone."

		# ask the user for confirmation to delete
		reply = QMessageBox.question(
			self, 
			title,
			text,
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
		)

		# Return if cancelled
		if reply == QMessageBox.StandardButton.No:
			return

		# -------------------- Main Action: Delete entries --------------------

		# Delete entries on database
		try:
			with self.dbConn.cursor() as cursor:
				seperator = ","
				pieces = ["%s"] * len(selectedIDs)
				placeholder = seperator.join(pieces) # example: "%s, %s, %s"
				sql = f"DELETE FROM jobs WHERE id IN ({placeholder})"
				cursor.execute(sql, selectedIDs)
				self.dbConn.commit()
		except Exception as e:
			self.dbConn.rollback()
			error(title="Database Error", 
				msg=f"Database Error in DatabaseTab.sqlDeleteActions():\n\n{e}", 
				terminate=True)
		
		# ML: mark deleted items as dirty and increment change counter x times
		self.MLmodel.markDirtyIDs(selectedIDs)
		self.MLmodel.incrementDbChangeCounter(len(selectedIDs))

		# Signal DB Change to Dashboard Tab
		self.dashboardTab.reRender = True

		# Re render table for recent changes
		self.renderTable()

	# **************************** EVENT HANDLERS ****************************

	def toggleDeleteButtonClickability(self):
		"""
		Allow clickability of delete button if there are entries selected. 
		Otherwise disallow clickability of the delete button.
		"""

		toggle = True if len(self.table.selectedIndexes()) > 0 else False
		self.deleteButton.setEnabled(toggle)

	def handleSaveToCsv(self):
		"""
		Saves the current table of the database tab to CSV file.

		Here are the steps that it takes:
		1. Dialog the filepath
		2. Call on sqlSelectActions to get the current view of the database 
			(with sorts and filter)
		3. Create a file and write the entries.
		"""

		# ----------------------- Get the csv filepath -----------------------

		filepath, _ = QFileDialog.getSaveFileName(self,
						"Save To CSV",
						"",
						"CSV Files (*.csv);;All Files (*)"
					)
		
		# return if canceled 
		if (filepath == ""):
			return

		# --------- Fetch table's data from sql and ML, write to file ---------

		with open(filepath, "w") as f:
			# Write the header
			f.write("Date,Job Name,Job Type,Hours Worked,Pay,Expected Pay\n")

			# fetch the entries based on filters and sorts from db
			data = self.sqlSelectActions()

			# Ask ML for predictions
			ids = [tup[0] for tup in data] # 0th index of each tuple is id
			predictedPayList = self.MLmodel.predict(ids)

			for i in range(len(data)):
				# index 1: job date
				jobDate = data[i][1] # type of datetime.date
				# Reformat to something like "Jun 01, 2025" in string format
				jobDateFormattedStr = jobDate.strftime("%b %d, %Y")
				f.write(f"\"{jobDateFormattedStr}\",")

				# index 2: job name
				jobName = data[i][2]
				jobName = f"\"{jobName}\"" if "," in jobName else jobName
				f.write(f"{jobName},")

				# index 3: job type
				jobType = data[i][3]
				f.write(f"{jobType},")

				# index 4: hours worked
				hoursWorked = data[i][4]
				f.write(f"{hoursWorked:.2f},")

				# index 5: pay
				pay = data[i][5]
				f.write(f"{pay:,.2f},")

				# From ML prediction: predicted pay
				id = data[i][0]
				predictedPay = predictedPayList[id]
				f.write(f"{predictedPay:,.2f}\n")

class MaintenanceTab(QWidget):
	"""
	The Maintenance Tab purposes are to:
	- Populate the database from CSV.
	- Reset database by deleting all entries.
	"""
	
	# ***************************** GUI and INITS *****************************

	def __init__(self, dbConn, MLmodel, dashboardTab):
		super().__init__()
		
		# ------------------------ Field Declarations ------------------------

		# Backend fields
		self.dbConn = dbConn
		self.MLmodel = MLmodel
		self.dashboardTab = dashboardTab # used to signal re render

		# UI Fields
		self.layout = QVBoxLayout()
		self.appendFromCsvButton = QPushButton("Append From CSV")
		self.clearDbButton = QPushButton("Clear Database")

		# --------------------------- Initialize UI ---------------------------

		self.initUI()
	
	def initUI(self):
		# Initialize layout
		self.setLayout(self.layout)

		# Initialize appendFromCsvButton
		self.appendFromCsvButton.clicked.connect(self.handleAppendFromCsv)
		self.appendFromCsvButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(self.appendFromCsvButton, alignment=Qt.AlignmentFlag.AlignHCenter)

		# Initialize clearDbButton
		self.clearDbButton.clicked.connect(self.handleClearDb)
		self.clearDbButton.setFocusPolicy(Qt.FocusPolicy.NoFocus)
		self.layout.addWidget(self.clearDbButton, alignment=Qt.AlignmentFlag.AlignHCenter)

	# **************************** EVENT HANDLERS ****************************

	def handleAppendFromCsv(self):
		"""
		Populate the database from CSV file.

		Here are the steps that it takes:
		1. Ask for the CSV filepath.
		2. Parse CSV into a pandas dataframe for easier formatting.
		3. Insert each row into the database.
		4. Signal to dashboard tab that db has been changed.
		"""
		# ----------------------- Get the csv filepath -----------------------

		filepath, _ = QFileDialog.getOpenFileName(
						self,
						"Select CSV File to Append", # caption
						"", # dir
						"CSV Files (*.csv);;All Files (*)" # filter
					)

		# --------------------------- Guard clause ---------------------------
		
		# return if canceled 
		if (filepath == ""):
			return

		# -------------------- Parse csv into a dataframe --------------------

		df = None
		try:
			df = pd.read_csv(
				filepath,
				parse_dates = ["Date"],
				converters = {
					"Job Type": lambda x: x.lower().strip(),
					"Hours Worked": lambda x: float(x),
					"Pay": lambda x: float(x.replace("$", "").replace(" ", "").replace(",", ""))
			})
			df["Date"] = df["Date"].dt.normalize().dt.strftime("%Y-%m-%d")
		except Exception as e:
			error(title="CSV Read Error",
		 		msg=f"CSV Read Error:\n\n{e}")
			return

		# -------------------- Enforce strict header names --------------------

		expectedHeader1 = ["Date", "Job Name", "Job Type", "Hours Worked", "Pay"]
		expectedHeader2 = ["Date", "Job Name", "Job Type", "Hours Worked", "Pay", "Expected Pay"]
		if list(df.columns) != expectedHeader1 and list(df.columns) != expectedHeader2:
			error(title="CSV Read Error",
		 		msg=f"CSV Read Error:\n\nHeaders are expected to be\n{expectedHeader1}\nor\n{expectedHeader2}"
				)
			return
			
		# ----------------- Iterate entries and insert to db  -----------------

		for _, row in df.iterrows():
			try:
				with self.dbConn.cursor() as cursor:
					sql =   """
							INSERT INTO jobs (job_date, job_name, job_type, hours_worked, pay)
							VALUES (%s, %s, %s, %s, %s)
							"""             # same as sql command
					values = (row["Date"], row["Job Name"], row["Job Type"], row["Hours Worked"], row["Pay"])
					cursor.execute(sql, values)
					self.dbConn.commit()
			except Exception as e:
				self.dbConn.rollback()
				error(title="Database Error", 
					msg=f"Database Error in MaintenanceTab.handleAppendFromCsv():\n\n{e}", 
					terminate=True)
		
		# ------------ Signal to dashboard tab that db is changed ------------

		self.dashboardTab.reRender = True
		
		# ------------------- Display Confirmation Message -------------------

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Icon.Information)
		msgBox.setWindowTitle("Success")
		msgBox.setText("Data Successfully Appended From CSV")
		msgBox.exec()
	
	def handleClearDb(self):
		"""
		Deletes all entries in the database.

		1. Checks if database is already empty, no point to delete an empty one.
		2. Comfirm with user for database deletion.
		3. Delete database entries.
		4. Reset ML, this includes deleting everything that has to do with old 
			model.
		5. Signal to dashboard tab that db has been changed.
		"""

		# -------------- Guard Clause: Do not delete on empty db --------------

		# Get the count of entries in db
		count = 0
		try: 
			with self.dbConn.cursor() as cursor:
				cursor.execute("SELECT COUNT(*) FROM jobs")
				result = cursor.fetchall()
				count = result[0][0]
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in MaintenanceTab.handleClearDb():\n\n{e}", 
				terminate=True)
		
		# Display already empty message
		if count == 0:
			error(title="Clear Database Warning", 
				msg="Database is already empty!!!")
			return

		# --------------- Display Delete Confirmation Question ---------------

		reply = QMessageBox.question(
			self, 
			"Confirm Clear Database", # title
			"Are you sure you want to delete all entries in the database?\n\nThis action cannot be undone.",
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
		)

		if reply == QMessageBox.StandardButton.No:
			return

		# -------------------- Main action: delete entries --------------------

		# reset DB try:
		try:
			with self.dbConn.cursor() as cursor:
				cursor.execute("DELETE FROM jobs")
				cursor.execute("ALTER TABLE jobs AUTO_INCREMENT = 1")
				self.dbConn.commit()
		except Exception as e:
			self.dbConn.rollback()
			error(title="Database Error", 
				msg=f"Database Error in MaintenanceTab.handleClearDb():\n\n{e}", 
				terminate=True)
		
		# reset ML
		self.MLmodel.reset()

		# Signal to dashboard tab that db is changed
		self.dashboardTab.reRender = True

		# ------------------- Display Confirmation Message -------------------

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Icon.Information)
		msgBox.setWindowTitle("Success")
		msgBox.setText("Database Successfully Cleared")
		msgBox.exec()
	
class MainWindow(QMainWindow):
	"""
	MainWindow is a class that manages all the other tabs (DashboardTab, 
	DatabaseTab, MaintenanceTab) and the shared backend references such as 
	dbConn and MLmodel.
	"""
	def __init__(self):
		super().__init__()
		
		# ------------------------ Field Declarations ------------------------

		# Backend fields
		self.dbConn = None # shared reference of database connection
		self.MLmodel = None # shared reference to the ML model

		# UI fields
		self.tabs = QTabWidget()
		self.dashboardTab = None
		self.databaseTab = None
		self.maintenanceTab = None

		# -------------------------- Initial Actions --------------------------
		
		# Initialize self.dbConn
		try:
			self.dbConn = mysql.connector.connect(**config)
		except Exception as e:
			error(title="Database Error", 
				msg=f"Database Error in MainWindow.__init__():\n\n{e}", 
				terminate=True)
		
		# Initialize self.MLmodel
		self.MLmodel = ML_MODEL(self.dbConn)
		
		# Initialize UI
		self.initUI()

	def initUI(self):
		# main window css
		mainWindowStyleSheet = """"""

		# Set MainWindow Properties
		self.setStyleSheet(mainWindowStyleSheet)
		self.setWindowTitle("Commission Work Accounting")
		self.setMinimumWidth(1250)
		self.setMinimumHeight(670)

		# Configure the tabs / new pages
		self.dashboardTab = DashboardTab(self.dbConn) 
		self.databaseTab = DatabaseTab(self.dbConn, self.MLmodel, self.dashboardTab)
		self.maintenanceTab = MaintenanceTab(self.dbConn, self.MLmodel, self.dashboardTab)
		self.tabs.addTab(self.dashboardTab, "Dashboard")
		self.tabs.addTab(self.databaseTab, "Database")
		self.tabs.addTab(self.maintenanceTab, "Maintenance")
		self.tabs.currentChanged.connect(self.handleTabChanged)
		self.setCentralWidget(self.tabs)
	
	def handleTabChanged(self, tabIndex):
		"""
		Renders important components of a tab when changed.
		"""
		if tabIndex == 0:
			# print("on dashboard tab")
			if self.dashboardTab.reRender:
				self.dashboardTab.render()
		elif tabIndex == 1:
			# print("on database tab")
			self.databaseTab.renderTable()
		elif tabIndex == 2:
			# print("on maintenance tab")
			pass
		
	def closeEvent(self, event):
		"""
		Closes the database connection. 
		"""
		self.dbConn.close() # close the connection when program is closed
		event.accept()

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = MainWindow()
	window.show()
	sys.exit(app.exec())