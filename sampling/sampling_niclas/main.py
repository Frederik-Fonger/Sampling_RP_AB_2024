import sampling_n as s
import pandas as pd







if __name__ == "__main__":
   
  
    
    # Test für CSV: 

    # Default format_timestamp_conversion_to_datetime_obj = 'YYYY-MM-DD HH:MM:SS.ssssss'
    remainder_plus_sample = s.remainder_plus_sampling(dataframe_or_filepath_to_log="Eventlogs/running-example.csv", 
                                                      format="CSV", sample_ratio=0.5, 
                                                      case_id_column="Case ID", 
                                                      activity_column="Activity", 
                                                      timestamp_column= "dd-MM-yyyy:HH.mm", 
                                                      format_timestamp_conversion_to_datetime_obj="%d-%m-%Y:%H.%M", 
                                                      min_columns=True)
    print(remainder_plus_sample)
    

    
    # # Test für Dataframe: 
    
    # dataframe = pd.read_csv("Eventlogs/running-example.csv", sep=";")

    # remainder_plus_sample = s.remainder_plus_sampling(dataframe_or_filepath_to_log=dataframe, 
    #                                                   format="DF", 
    #                                                   sample_ratio=0.5, 
    #                                                   case_id_column="Case ID", 
    #                                                   activity_column="Activity", 
    #                                                   timestamp_column= "dd-MM-yyyy:HH.mm", 
    #                                                   format_timestamp_conversion_to_datetime_obj="%d-%m-%Y:%H.%M", 
    #                                                   min_columns=False)
    # print(remainder_plus_sample)



    
    # # Test für XES:

    # remainder_plus_sample = s.remainder_plus_sampling(dataframe_or_filepath_to_log="Eventlogs/Sepsislog.xes", 
    #                                                   format="XES", 
    #                                                   sample_ratio=0.5, 
    #                                                   case_id_column="case:concept:name", 
    #                                                   activity_column="concept:name", 
    #                                                   timestamp_column= "time:timestamp", 
    #                                                   format_timestamp_conversion_to_datetime_obj="YYYY-MM-DD HH:MM:SS.ssssss", 
    #                                                   min_columns=False)
    # print(remainder_plus_sample)


    # allbehaviour_sample = s.allbehaviour_Sampling(dataframe_or_filepath_to_log="Eventlogs/Sepsislog.xes", 
    #                                               format="XES", 
    #                                               sample_ratio=0.04, 
    #                                               case_id_column="", 
    #                                               activity_column="", 
    #                                               timestamp_column="", 
    #                                               min_columns=True)
    # print(allbehaviour_sample)













