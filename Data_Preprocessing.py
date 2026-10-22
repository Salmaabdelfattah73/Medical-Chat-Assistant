# ============================================================
# Medical Q&A - PySpark Preprocessing Pipeline
# Course: CISC 886 - Cloud Computing, Queen's University
# Runs on AWS EMR - reads from S3, writes to S3
# ============================================================

# Install matplotlib in a temporary location to avoid system conflicts

import subprocess, sys
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--target", "/tmp/pylibs",
    "--upgrade", "python-dateutil", "matplotlib","pandas"
])
sys.path.insert(0, "/tmp/pylibs")

# --- Step 1: Initialize Spark Session ---
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Medical_QA_Preprocessing") \
    .getOrCreate()

print("Spark session created")

# --- Step 2: Import all PySpark functions ---
from pyspark.sql.functions import (
    col, concat, lit, when, lower, trim,
    split, size, length, regexp_replace, element_at
)

# --- Step 3: Load Raw Data from S3 ---
S3_INPUT = "s3://25cdkg-medical-qa/25cdkg-raw-data/train.csv"
S3_OUTPUT = "s3://25cdkg-medical-qa/25cdkg-processed-data/"
S3_EDA = "s3://25cdkg-medical-qa/25cdkg-eda/"

df = spark.read.csv(S3_INPUT, header=True, inferSchema=True)

print(f"Data loaded - {df.count()} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns}")
df.select("qtype", "Question", "Answer").show(3, truncate=False)

# --- Step 4: Data Cleaning & Transformation ---
df_clean = df.withColumn("Question", lower(trim(col("Question")))) \
             .withColumn("Answer", lower(trim(col("Answer")))) \
             .withColumn("qtype", lower(trim(col("qtype"))))

df_clean = df_clean.dropna(subset=["Question", "Answer"])
print(f"After dropping nulls: {df_clean.count()} rows")

# --- Step 5: Feature Engineering (Word Counts) ---
df_eda = df_clean.withColumn("q_len", size(split(col("Question"), " "))) \
                 .withColumn("a_len", size(split(col("Answer"), " ")))

print("Summary Statistics for Token/Word Counts:")
df_eda.select("q_len", "a_len").describe().show()

# --- Step 6: EDA Figures ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

pdf = df_eda.select("q_len", "a_len", "qtype").sample(fraction=0.3, seed=42).toPandas()

# Figure 1: Question Length Distribution
plt.figure(figsize=(8, 4))
pdf['q_len'].hist(bins=30, color='#4CAF50', edgecolor='black')
plt.title('Figure 1: Distribution of Question Word Counts')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("/tmp/fig1_question_length.png", dpi=150)
plt.close()
print("Figure 1 saved")

# Figure 2: Top 10 Question Types
plt.figure(figsize=(10, 5))
pdf['qtype'].value_counts().head(10).plot(kind='bar', color='#2196F3')
plt.title('Figure 2: Top 10 Question Types (Label Balance)')
plt.xlabel('Question Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("/tmp/fig2_label_balance.png", dpi=150)
plt.close()
print("Figure 2 saved")

# Figure 3: Q vs A Length
plt.figure(figsize=(8, 4))
plt.scatter(pdf['q_len'], pdf['a_len'], alpha=0.5, color='#FF5722')
plt.title('Figure 3: Question vs Answer Length Correlation')
plt.xlabel('Question Word Count')
plt.ylabel('Answer Word Count')
plt.tight_layout()
plt.savefig("/tmp/fig3_q_vs_a_scatter.png", dpi=150)
plt.close()
print("Figure 3 saved")

# Figure 4: Top 5 Categories Pie
df_for_qtype_plot = df_eda.withColumn("qtype_clean",
    when(col("qtype").contains(":"),
         trim(element_at(split(col("qtype"), ":"), -1)))
    .otherwise(col("qtype")))

df_for_qtype_plot = df_for_qtype_plot.filter(
    (col("a_len") >= 10) & (col("a_len") <= 500) & (col("q_len") >= 5))

qtype_counts = df_for_qtype_plot.groupBy("qtype_clean").count() \
    .orderBy("count", ascending=False).limit(5).toPandas()

plt.figure(figsize=(8, 8))
plt.pie(qtype_counts['count'], labels=qtype_counts['qtype_clean'],
        autopct='%1.1f%%', startangle=140,
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
plt.title("Figure 4: Distribution of Top 5 Medical Question Types")
plt.tight_layout()
plt.savefig("/tmp/fig4_top5_categories.png", dpi=150)
plt.close()
print("Figure 4 saved")

# Upload figures to S3
for fig_file in ["fig1_question_length.png", "fig2_label_balance.png",
                 "fig3_q_vs_a_scatter.png", "fig4_top5_categories.png"]:
    os.system(f"aws s3 cp /tmp/{fig_file} {S3_EDA}{fig_file}")
print("All EDA figures uploaded to S3")

# --- Step 7: Clean the qtype column ---
df_refined = df_eda.withColumn("qtype_clean",
    when(col("qtype").contains(":"),
         trim(element_at(split(col("qtype"), ":"), -1)))
    .otherwise(col("qtype")))

# --- Step 8: Outlier Analysis ---
print("=== Answer Length Statistics ===")
df_eda.select(col("a_len").alias("answer_words")).describe().show()

total_rows = df_eda.count()
rows_gt_500 = df_eda.filter(col('a_len') > 500).count()
rows_gt_250 = df_eda.filter(col('a_len') > 250).count()
print(f"Rows > 500 words: {rows_gt_500} ({(rows_gt_500/total_rows)*100:.1f}%)")
print(f"Rows > 250 words: {rows_gt_250} ({(rows_gt_250/total_rows)*100:.1f}%)")

# --- Step 9: Outlier Removal ---
df_filtered = df_refined.filter(
    (col("a_len") >= 10) & (col("a_len") <= 500) & (col("q_len") >= 5))
print(f"After outlier removal: {df_filtered.count()} rows")

# --- Step 10: Final Cleanup + Deduplication ---
df_final_clean = df_filtered \
    .withColumn("Answer", regexp_replace(col("Answer"), r"\s+", " ")) \
    .withColumn("Question", regexp_replace(col("Question"), r"\s+", " ")) \
    .dropDuplicates(["Question", "Answer"])
print(f"After dedup: {df_final_clean.count()} rows")

# --- Step 11: ChatML Formatting ---
df_templated = df_final_clean.withColumn("text",
    concat(
        lit("<|im_start|>system\nYou are a professional medical assistant. "
            "Answer the patient's questions accurately.<|im_end|>\n"),
        lit("<|im_start|>user\n"), col("Question"), lit("<|im_end|>\n"),
        lit("<|im_start|>assistant\n"), col("Answer"), lit("<|im_end|>")
    ))

df_final = df_templated.filter(
    (length(col("Question")) > 20) & (length(col("Answer")) > 40))
print(f"Final dataset: {df_final.count()} rows")

# --- Step 12: Train/Val/Test Split ---
train_df, val_df, test_df = df_final.randomSplit([0.7, 0.15, 0.15], seed=42)
print(f"Train: {train_df.count()}")
print(f"Validation: {val_df.count()}")
print(f"Test: {test_df.count()}")

# Figure 5: Split Counts
split_counts = {
    "Train": train_df.count(),
    "Validation": val_df.count(),
    "Test": test_df.count()
}
plt.figure(figsize=(6, 4))
plt.bar(split_counts.keys(), split_counts.values(),
        color=["#4CAF50", "#2196F3", "#FF5722"])
plt.title("Figure 5: Sample Count Per Split")
plt.ylabel("Number of Samples")
for i, (k, v) in enumerate(split_counts.items()):
    plt.text(i, v + 50, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("/tmp/fig5_split_counts.png", dpi=150)
plt.close()
os.system(f"aws s3 cp /tmp/fig5_split_counts.png {S3_EDA}fig5_split_counts.png")
print("Figure 5 saved")

# --- Step 13: Save to S3 ---
train_df.select("text").coalesce(1).write.mode("overwrite") \
    .json(S3_OUTPUT + "train")
val_df.select("text").coalesce(1).write.mode("overwrite") \
    .json(S3_OUTPUT + "validation")
test_df.select("text").coalesce(1).write.mode("overwrite") \
    .json(S3_OUTPUT + "test")
test_df.select("Question", "Answer").coalesce(1).write.mode("overwrite") \
    .json(S3_OUTPUT + "test_qa")
print("All output files saved to S3")

# --- Step 14: Stop Spark ---
spark.stop()
print("Pipeline complete!")