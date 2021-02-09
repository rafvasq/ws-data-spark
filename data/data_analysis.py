from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from math import radians, cos, sin, asin, sqrt, pi, pow


def read_data():
    """
    Read in provided data
    :return: Spark DF of requests and POI data
    """
    requests = spark.read.option("header", True) \
        .option("inferSchema", True) \
        .option("ignoreLeadingWhiteSpace", True) \
        .csv("/tmp/data/DataSample.csv") \
        .withColumnRenamed('_ID', 'request_id') \
        .withColumnRenamed('Latitude', 'request_latitude') \
        .withColumnRenamed('Longitude', 'request_longitude')

    pois = spark.read.option("header", True) \
        .option('inferSchema', True) \
        .option("ignoreLeadingWhiteSpace", True) \
        .csv('/tmp/data/POIList.csv') \
        .withColumnRenamed('Latitude', 'poi_latitude') \
        .withColumnRenamed('Longitude', 'poi_longitude') \
        .dropDuplicates(['poi_latitude', 'poi_longitude'])

    return requests, pois


def clean_data(requests):
    """
    Filters out suspicious request records
    :param requests: Spark DF of requests
    :return: Cleansed Spark DF of requests
    """
    cleaned = requests.dropDuplicates(
        ['TimeSt', 'Country', 'Province', 'City', 'request_latitude', 'request_longitude'])
    return cleaned


def get_distance(long_a, lat_a, long_b, lat_b):
    """
    Calculates distance using Harversine formula
    :param long_a: Point A Longitude
    :param lat_a: Point A Latitude
    :param long_b: Point B Longitude
    :param lat_b: Point B Latitude
    :return: Distance between Point A and Point B
    """
    long_a, lat_a, long_b, lat_b = map(radians, [long_a, lat_a, long_b, lat_b])
    dist_long = long_b - long_a
    dist_lat = lat_b - lat_a
    area = sin(dist_lat / 2) ** 2 + cos(lat_a) * cos(lat_b) * sin(dist_long / 2) ** 2
    central_angle = 2 * asin(sqrt(area))
    radius = 6371
    distance = abs(central_angle * radius)

    return distance


def label_data(requests, pois):
    """
    Assigns each request to the closest POI
    :param requests: Spark DF of requests
    :param pois: Spark DF of POIs
    :return: Spark DF containing request and POI information
    Schema[request_id, distance, POIID, poi_latitude, poi_longitude, request_longitude, request_latitude]
    """
    udf_distance = F.udf(get_distance)
    crossed = pois.crossJoin(requests.select(["request_id", "request_longitude", "request_latitude"]))
    crossed = crossed.withColumn("distance",
                                 udf_distance(
                                     crossed.poi_longitude, crossed.poi_latitude,
                                     crossed.request_longitude, crossed.request_latitude).cast(DoubleType()))
    labelled = (
        crossed.select(["POIID", "request_id", "distance"])
            .groupBy("request_id")
            .agg(F.min('distance').alias("distance"))
            .join(crossed, on=["request_id", "distance"])
    )

    return labelled


def get_density(radius, count_requests):
    """
    Calculates the density of requests within a circle's area
    :param radius: Radius of the cirlce
    :param count_requests: Number of requests
    :return: Density
    """
    area = pi * radius ** 2
    density = count_requests / area
    return density


def analyze_data(labelled):
    """
    Compiles statistics about the data
    :param labelled: Spark DF of the labelled data
    :return: Spark DF with various statistics of each POI
    Schema[POIID, stddev_distance, mean_distance, radius, count_requests, density]
    """
    density_udf = F.udf(get_density)

    analyzed = \
        labelled.groupby('POIID') \
            .agg(F.stddev('distance').alias("stddev_distance"),
                 F.mean('distance').alias("mean_distance"),
                 F.max("distance").alias("radius"),
                 F.count("distance").alias("count_requests"))

    analyzed = analyzed.withColumn('density', density_udf(analyzed.radius, analyzed.count_requests))

    return analyzed


def get_popularity(stddev_distance, mean_distance, count_requests, total_requests):
    """
    Calculates the "popularity" of a POI using a mathematical model which considers the distances of requests to a POI
    in the form of the coefficient of variation (stddev relative to the mean) and the proportion of all requests a POI
    has. Bounded between -10 and 10, it behaves as a logistic function (sigmoid), penalizing the score if there is more
    variability in the distances between requests and a POI (the coefficient of variation is larger) and for having a
    smaller share of the total requests (proportion).
    :param stddev_distance: Standard deviation of the distances between a POI and its requests
    :param mean_distance: Mean distance between a POI and its requests
    :param count_requests: The count of a POI's requests
    :param total_requests: The count of all of the requests
    :return: The popularity score
    """
    coefficient = stddev_distance / mean_distance
    proportion = count_requests / total_requests

    x = (1 / pow(coefficient, 2)) - (1 - (proportion))
    popularity = 10 * (x / sqrt(1 + pow(x, 2)))

    return popularity


def model_data(analyzed, requests):
    """
    Assigns each POI a popularity score
    :param analyzed: Spark DF with various statistics of each POI
    :param requests: Spark DF containing cleansed requests
    :return: Spark DF with various statistics including popularity score
    Schema[POIID, stddev_distance, mean_distance, radius, count_requests, density, popularity]
    """
    popularity_udf = F.udf(get_popularity)
    modelled = analyzed.withColumn("popularity",
                                   popularity_udf(analyzed.stddev_distance,
                                                  analyzed.mean_distance,
                                                  analyzed.count_requests,
                                                  F.lit(requests.count())))

    return modelled


if __name__ == "__main__":
    spark = SparkSession.builder.appName("EQWorks").getOrCreate()

    requests, pois = read_data()

    cleaned = clean_data(requests)

    labelled = label_data(requests, pois)

    analyzed = analyze_data(labelled)

    modelled = model_data(analyzed, cleaned)

    modelled.show()

    spark.stop()