# seirsplus-with-hyperopt
The model requires a series of optimisations parses executed with the optimisation MK2 .py programme
This model currently utilises PySpark, but it is likely PySpark will be removed from Molly (due to a decison made by Orbis)
If PySpark is removed then the SparkTrials() class will need to be replaced with the standard Trials() class
Once this is completed, comment out the following code:

  from pyspark.sql import SparkSession

  os.environ['PYSPARK_PYTHON'] = sys.executable
  os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
  spark = SparkSession.builder.getOrCreate()
  
This will force the model to re-run in a sequential manner, instead of the parallel fashion PySpark allows

Two major limitations of the model are that it assumes a constant proportion of asymopmatic adult aged indivduals
and a constant proportion of asymopmatic children. In reality the proportion of asymptomatic individuals likely varies with age.#

The second major limitation, is that this model is built for a sample population based on the demographic structure of Surrey but smaller.
As such the values are extrapolated to represent the final population size of Surrey.
This therefore assumes a consistent population density.

The optimisation parse relies on the past 13 R values (fortnightly). These can be entered into the list called R_Values

Here all you need to worry about is the numerical decimal, the rest takes care of itself, with N being the sample size designated
at the start of the script.

CasesGovUk and DeathsGovUk are manually updated from the gov UK api, automatic data retrieval will be the next stage

All parameters housed in the search_space are dealt with during the optimisation parses, they have upper an lower limits within which
values are taken.

After the optimisation parses have been completed take the values from the BestParameters.txt file and enter into the final model

'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'P_GLOB', 'p_asym' and 'betaMUL' relate directly to their namesakes 

'hDP_c': hospitalizationToDischargePeriod_coeffvar
'hDP_m': hospitalizationToDischargePeriod_mean
'hDeP_c': hospitalizationToDeathPeriod_coeffvar
'hDeP_m': hospitalizationToDeathPeriod_mean
'lP_c': latentPeriod_coeffvar
'lP_m': latentPeriod_mean
'oHP_c': onsetToHospitalizationPeriod_coeffvar
'oHP_m': onsetToHospitalizationPeriod_mean
'psP_c': presymptomaticPeriod_coeffvar
'psP_m': presymptomaticPeriod_mean
'r0_c': R1_coeffvar
'sP_c': symptomaticPeriod_coeffvar
'sP_m': symptomaticPeriod_mean

R0_mean is equal to the R6 value in the optimisation parse