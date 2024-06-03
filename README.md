# Sampling

In this project we provide the code for the Remainder Plus and AllBehavior event log sampling algorithm. Additionally, we provide the code and event logs for the evaluation of the resulting samples.


Code:
	
	We used Python 3.10 with pip 22.3.1 on Windows 11. Further package requirements can be found in requirements.txt.
	
	The python code is in the folder "/sampling".
	
	Starting sampling with evaluation:
		Starting the main.py will start the sampling and evaluation for one event log. Event logs and further parameters can be set in the ConfigManager.py.
		The samples and the evaluation results are saved as files. 
		
		For multiprocessing, start the multi_p.py. The same function as in the main.py will be started but with many event logs in parallel. For each event log one process instance will be started. 

		

Evaluation results:
	
	The Evaluation results are exported as excel-files and plotted graphs in the folder "evaluation_results". The results for the Earth mover´s distance (emd) are seperate, as for the emd only 9 of the 10 event logs computed in a reasonoable amount of time.
	
	Used event logs: 
	
		BPI-Challenge 2012
		BPI-Challenge 2015
		Domestic Declaration Log
		International Declaration Log
		Permit Log
		Prepaid Travel Cost Log
		Request for Payment Log
		Sepsis Log
		CoSeLoG WABO 2
		CoSeLoG WABO 3

	Used quality measures for the samples:
	
		Eath mover´s distance
		coverage
		MAE
		NMAE
		RMSE
		NRMSE
		MAPE
		mean sample ratio
		standard deviation
		percentage of unsampled DFRs
		percentage of undersampled DFRs
		percentage of oversampled DFRs
		percentage of truly sampled DFRs
		percentage of DFRs within 1 % of ideal
		percentage of DFRs within 5 % of ideal
		percentage of DFRs within 10 % of ideal