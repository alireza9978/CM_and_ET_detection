# persien font
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from spark_test.functions import *


def prepare_for_kmeans(temp_sdf):
    temp = temp_sdf

    # define function for split power column
    def split_power_columns(inner_temp_sdf):
        return inner_temp_sdf.select("#", "V", "N", "date", "id", "uid", inner_temp_sdf.power[0].alias("H0"),
                                     inner_temp_sdf.power[1].alias("H1"), inner_temp_sdf.power[2].alias("H2"),
                                     inner_temp_sdf.power[3].alias("H3"), inner_temp_sdf.power[4].alias("H4"),
                                     inner_temp_sdf.power[5].alias("H5"), inner_temp_sdf.power[6].alias("H6"),
                                     inner_temp_sdf.power[7].alias("H7"), inner_temp_sdf.power[8].alias("H8"),
                                     inner_temp_sdf.power[9].alias("H9"), inner_temp_sdf.power[10].alias("H10"),
                                     inner_temp_sdf.power[11].alias("H11"), inner_temp_sdf.power[12].alias("H12"),
                                     inner_temp_sdf.power[13].alias("H13"), inner_temp_sdf.power[14].alias("H14"),
                                     inner_temp_sdf.power[15].alias("H15"), inner_temp_sdf.power[16].alias("H16"),
                                     inner_temp_sdf.power[17].alias("H17"), inner_temp_sdf.power[18].alias("H18"),
                                     inner_temp_sdf.power[19].alias("H19"), inner_temp_sdf.power[20].alias("H20"),
                                     inner_temp_sdf.power[21].alias("H21"), inner_temp_sdf.power[22].alias("H22"),
                                     inner_temp_sdf.power[23].alias("H23"), inner_temp_sdf.statistics[0].alias("S0"),
                                     inner_temp_sdf.statistics[1].alias("S1"), inner_temp_sdf.statistics[2].alias("S2"),
                                     inner_temp_sdf.statistics[3].alias("S3"))

    # call the split_power function
    temp = split_power_columns(temp)

    # filter date
    # temp=temp.filter(temp.date > "2014-08-15").filter(temp.date < "2014-08-19") #filter dates
    # temp=temp.filter(temp.id == "Apt40") #filter IDs
    temp = temp.filter(temp.V)  # filter valid rows

    FEATURES = ['H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14', 'H15',
                'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'S0', 'S1', 'S2', 'S3']

    # call the generate_uniqe_id function
    temp = generate_unique_id(temp)

    # make ready
    vecAssembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
    # df_kmeans = vecAssembler.transform(temp).select(col("uid").alias("id"), col("features"))
    df_kmeans = vecAssembler.transform(temp).select(col("uid"), col("features"))
    return df_kmeans


def kmeans(temp_sdf):
    # find best k
    MAX_k = 8
    costs = np.zeros(MAX_k)
    silhouettes = np.zeros(MAX_k)
    silhouettes[1] = 0  # set value for k=1
    for k in range(2, MAX_k):
        kmeans_model = KMeans().setK(k).setSeed(1)
        model = kmeans_model.fit(temp_sdf)
        costs[k] = model.computeCost(temp_sdf)  # requires Spark 2.0 or later
        predictions = model.transform(temp_sdf)
        evaluator = ClusteringEvaluator()
        silhouettes[k] = evaluator.evaluate(predictions)

    # show silhouette
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, MAX_k), silhouettes[2:MAX_k])
    ax.set_xlabel('k')
    ax.set_ylabel('silhouette')

    # show cost
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, MAX_k), costs[2:MAX_k])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')

    # find best k
    best_k = np.argmax(silhouettes)
    print("maximum value of silhouette is: " + str(silhouettes[best_k]) + " in index: " + str(best_k))

    # Trains a k-means model.
    kmeans_model = KMeans().setK(best_k).setSeed(1)
    model = kmeans_model.fit(temp_sdf)

    # Make predictions
    predictions = model.transform(temp_sdf)

    # Evaluate clustering by computing Silhouette score
    evaluator = ClusteringEvaluator()

    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    transformed = model.transform(temp_sdf).select('uid', 'prediction')
    transformed.show()
    transformed.groupby('prediction').count().show()
    rows = transformed.collect()
    prediction = spark.createDataFrame(rows)
    prediction.show()

    return model, best_k, silhouette  # silhouettes: new


def call_kmeans(temp_sdf):
    # create statistics dataframe
    kmeans_statistics_schema = StructType([StructField("id", StringType()),
                                           StructField("k", IntegerType()),
                                           StructField("Silhouette", FloatType())])

    temp_kmeans_statistics = spark.createDataFrame([], kmeans_statistics_schema)

    id_list = get_ids(temp_sdf)
    # replace sdf with final_sdf for clustering benign and malicious data
    sdf_kmeans = prepare_for_kmeans(temp_sdf)
    # sdf_kmeans=pca_for_kmeans(sdf_kmeans) #0.8725788926917551 to 0.9101118371931005
    # sdf_kmeans.show()
    iteration = 1
    for i in np.nditer(id_list):
        sdf_kmeans_by_id = sdf_kmeans.filter(sdf_kmeans.uid.like(str(i) + "-" + "%"))  # filter IDs
        print("customer " + str(iteration) + ": " + str(i))
        # sdf_kmeans_by_id.show()
        kmeans_model, best_k, silhouette = kmeans(sdf_kmeans_by_id)
        # kmeans_model.save(os.path.join(KMEANS_PATH,str(i)))
        summary = kmeans_model.summary
        if summary.clusterSizes[1] > 200:
            print("cluster size bigger than 200")
        else:
            print("cluster size smaller than 200")

        newRow_for_statistics = spark.createDataFrame([(str(i), int(best_k), float(silhouette))])
        temp_kmeans_statistics = temp_kmeans_statistics.union(newRow_for_statistics)

        iteration += 1
        # model_name = KMeansModel.load(os.path.join(KMEANS_PATH,str(i)) #for load model
    return temp_kmeans_statistics


spark = SparkSession.builder.appName("anomaly_detection").getOrCreate()

# load sdf
print("-------------------- loading data set!")
df = pd.read_pickle("my_data/iran*.pkl")
sdf = spark.createDataFrame(df)

print("-------------------- k-means started!")
kmeans_statistics = call_kmeans(sdf)
kmeans_statistics.show()

# save
result_pdf = kmeans_statistics.select("*").toPandas()
result_pdf.to_pickle('my_data/kmeans_statistics.pkl')

# load
# df = pd.read_pickle(os.path.join(BASE_PATH, 'kmeans_statistics.pkl'))
# df.head()
# df.describe()
