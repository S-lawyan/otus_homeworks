import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, median, collect_list, concat_ws, split, expr
from pyspark.sql.types import StringType
from pyspark.sql.functions import desc, asc
from pyspark.sql.window import Window
import pyspark.sql.functions as F

INPUT_CRIME_FILE = "in/crime.csv"
INPUT_OFFENSE_CODES_FILE = "in/offense_codes.csv"
OUTPUT_FILE = "out/hw_boston_crimes_analysis.parquet"

# Создание SparkSession
spark = (
    SparkSession.builder.appName("OTUS_HW_BostonCrimesAnalysis")
    .config("spark.some.config.option", "some-value")
    .config("spark.driver.memory", "8g")  # Увеличиваем память драйвера до 4 ГБ
    .config("spark.executor.memory", "8g")  # Увеличиваем память исполнителя до 4 ГБ
    .config("spark.memory.fraction", "0.9")  # Увеличиваем долю памяти для хранения данных
    .config("spark.memory.storageFraction", "0.9")  # Настройка доли памяти для хранения
    .getOrCreate()
)

# Шаг 1: Загрузка данных о преступлениях
crimes_df = spark.read.csv(INPUT_CRIME_FILE, header=True, inferSchema=True)
all_rows = crimes_df.count()
print(f"Исходное количество записей: {all_rows}")

# 2.2 Удаление дубликатов
crimes_df = crimes_df.dropDuplicates()
all_rows_without_duplicates = crimes_df.count()
print(f"Количество записей после удаления дубликатов: {all_rows_without_duplicates}")
print(f"Количество дубликатов: {all_rows - all_rows_without_duplicates}")
# crimes_df_dropped_duplicates.show(5)

# 2.3 Проверка на наличие null-значений
crimes_df = crimes_df.dropna(
    how='any',
    subset=["DISTRICT"]
)

all_rows_without_duplicates_and_null_values = crimes_df.count()
print(f"Количество записей после удаления null-значений: {all_rows_without_duplicates_and_null_values}")
print(f"Количество null-значений: {all_rows_without_duplicates - all_rows_without_duplicates_and_null_values}")
# print("DataFrame без дубликатов и null-значений: ")
# crimes_df.show(5)


# Шаг 2: Загрузка данных с кодами преступления
offense_codes_df = spark.read.csv(INPUT_OFFENSE_CODES_FILE, header=True, inferSchema=True)

# 2.1 Удаление дубликатов и выбор уникальных названий
offense_codes_df = offense_codes_df.dropDuplicates(["CODE"]).select("CODE", "NAME")
# print("DataFrame кодов преступления: ")
# offense_codes_df.show(5)


# Шаг 3:  Объединение crime_df с offense_codes_df для получения crime_type
joined_df = crimes_df.join(offense_codes_df,
                           crimes_df.OFFENSE_CODE == offense_codes_df.CODE,
                           "left")
# print("DataFrame после объединения: ")
# joined_df.show(5)


# 3.1 Получение первой части NAME как crime_type
joined_df = joined_df.withColumn("crime_type", split(col("NAME"), " ").getItem(0))
# print("DataFrame после преобразования crime_type: ")
# joined_df.show(5)


# Шаг 4: Построение витрины

# 4.1 Получение медианы количества преступлений по месяцам и типам
month_agg = joined_df.groupBy(["DISTRICT", "MONTH"]).agg(
    count("*").alias("CRIMES_MONTH_COUNT"),
).sort(desc("MONTH"))
medians = month_agg.groupBy(["DISTRICT", "MONTH"]).agg(
    expr("percentile_approx(CRIMES_MONTH_COUNT, 0.5)").alias("crimes_monthly_median"),
)
# medians.show(10)

# 4.2 Агрегация данных по районам c высислением средней широты и долготы
avg_lat_lon_df = joined_df.groupBy("DISTRICT").agg(
    expr("avg(Lat)").alias("Lat_avg"),  # СР.ЗН. по всем значениям широты
    expr("avg(Long)").alias("Long_avg")  # СР.ЗН. по всем значениям долготы
)
# avg_lat_lon_df.show(10)

# 4.3 Получение трех самых частых типов преступлений, конкотенация ТОП-3 через запятую
crimes_freq = joined_df.groupBy("DISTRICT", "crime_type").count()  # .orderBy(col("DISTRICT"), col("count").desc())
window_spec = Window.partitionBy("DISTRICT").orderBy(F.col("count").desc())
top_3_crimes_freq = crimes_freq.withColumn("rn", F.row_number().over(window_spec)) \
    .filter(F.col("rn") <= 3) \
    .groupBy("DISTRICT") \
    .agg(F.concat_ws(", ", F.collect_list("crime_type")).alias("frequent_crime_types")) \
    .orderBy("DISTRICT")

result_df = joined_df.join(medians, ["DISTRICT", "MONTH"], "inner")
result_df = result_df.join(avg_lat_lon_df, ["DISTRICT"], "inner")
result_df = result_df.join(top_3_crimes_freq, ["DISTRICT"], "inner")
# result_df.show(100)

# Шаг 5: Сохранение витрины в формате .parquet
result_df.write.parquet(OUTPUT_FILE, mode="overwrite")

# Завершение работы SparkSession
spark.stop()
