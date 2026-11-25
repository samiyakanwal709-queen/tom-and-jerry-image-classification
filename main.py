# ==========================================
# TOM AND JERRY IMAGE CLASSIFICATION
# EXPLORATORY DATA ANALYSIS DASHBOARD
# ==========================================

# 1. Import Libraries

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path


# ==========================================
# 2. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Tom & Jerry EDA Dashboard",
    page_icon="🐱",
    layout="wide"
)


# ==========================================
# 3. Dataset Path
# ==========================================

BASE_DIR = Path(__file__).parent

DATASET_PATH = BASE_DIR / "dataset"

categories = [
    "jerry",
    "tom",
    "tom_jerry_0",
    "tom_jerry_1"
]

image_extensions = [
    ".jpg",
    ".jpeg",
    ".png"
]


# ==========================================
# 4. Load Dataset Images
# ==========================================

image_data = []

for category in categories:

    folder_path = DATASET_PATH / category

    if folder_path.exists():

        for image_path in folder_path.rglob("*"):

            if image_path.suffix.lower() in image_extensions:

                image_data.append({
                    "Category": category,
                    "Image_Path": str(image_path),
                    "Image_Name": image_path.name
                })


# Convert Data into DataFrame

df = pd.DataFrame(image_data)


# ==========================================
# 5. Dashboard Title
# ==========================================

st.title("🐱 Tom & Jerry Image Classification")

st.markdown(
    "### Exploratory Data Analysis (EDA) Dashboard"
)

st.write(
    "This dashboard explores Tom and Jerry image categories "
    "and their image distribution."
)

st.divider()


# ==========================================
# 6. Sidebar
# ==========================================

st.sidebar.title("🐱 Tom & Jerry")

st.sidebar.markdown("### Image Analysis Project")

st.sidebar.write(
    "Welcome to our Image Classification Dashboard!"
)

# Tom Image

tom_images = df[df["Category"] == "tom"]

if len(tom_images) > 0:

    st.sidebar.subheader("Tom")

    st.sidebar.image(
        tom_images.iloc[0]["Image_Path"],
        use_container_width=True
    )

# Jerry Image

jerry_images = df[df["Category"] == "jerry"]

if len(jerry_images) > 0:

    st.sidebar.subheader("Jerry")

    st.sidebar.image(
        jerry_images.iloc[0]["Image_Path"],
        use_container_width=True
    )


# Sidebar Information

st.sidebar.divider()

st.sidebar.info(
    "Created for Exploratory Data Analysis"
)


# ==========================================
# 7. Dataset Validation
# ==========================================

if df.empty:

    st.error(
        "No images found! Please check your dataset folder."
    )

    st.stop()


# ==========================================
# 8. Total Images and Categories
# ==========================================

total_images = len(df)

total_categories = df["Category"].nunique()

average_images = np.mean(
    df["Category"].value_counts().values
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Images",
        total_images
    )

with col2:

    st.metric(
        "Total Categories",
        total_categories
    )

with col3:

    st.metric(
        "Average Images",
        round(average_images, 2)
    )


st.divider()


# ==========================================
# 9. Category Wise Image Count
# ==========================================

st.header("📊 Category Wise Image Distribution")

category_count = (
    df["Category"]
    .value_counts()
    .reindex(categories, fill_value=0)
)

count_df = category_count.reset_index()

count_df.columns = [
    "Category",
    "Total Images"
]


st.dataframe(
    count_df,
    use_container_width=True
)


# ==========================================
# 10. Matplotlib Bar Chart
# ==========================================

st.subheader("Matplotlib - Image Count")

fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.bar(
    category_count.index,
    category_count.values
)

ax1.set_title("Images in Each Category")

ax1.set_xlabel("Categories")

ax1.set_ylabel("Number of Images")

ax1.tick_params(axis="x", rotation=20)

plt.tight_layout()

st.pyplot(fig1)

plt.close(fig1)


# ==========================================
# 11. Seaborn Chart
# ==========================================

st.subheader("Seaborn - Category Distribution")

fig2, ax2 = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=count_df,
    x="Category",
    y="Total Images",
    ax=ax2
)

ax2.set_title("Category Wise Image Distribution")

ax2.tick_params(axis="x", rotation=20)

plt.tight_layout()

st.pyplot(fig2)

plt.close(fig2)


# ==========================================
# 12. Plotly Interactive Chart
# ==========================================

st.subheader("Plotly - Interactive Pie Chart")

fig3 = px.pie(
    count_df,
    names="Category",
    values="Total Images",
    title="Image Distribution by Category",
    hole=0.4
)

st.plotly_chart(
    fig3,
    use_container_width=True
)


# ==========================================
# 13. Category Selection
# ==========================================

st.divider()

st.header("🖼️ Explore Dataset Images")

selected_category = st.selectbox(
    "Select an Image Category",
    categories
)


# Filter Selected Category

filtered_df = df[
    df["Category"] == selected_category
]


st.write(
    "Total Images in Selected Category:",
    len(filtered_df)
)


# ==========================================
# 14. Display Sample Images
# ==========================================

st.subheader("Sample Images")

number_of_images = min(12, len(filtered_df))

for i in range(number_of_images):

    if i % 3 == 0:

        cols = st.columns(3)

    with cols[i % 3]:

        st.image(
            filtered_df.iloc[i]["Image_Path"],
            caption=filtered_df.iloc[i]["Image_Name"],
            use_container_width=True
        )


# ==========================================
# 15. Dataset Preview
# ==========================================

st.divider()

st.header("📁 Dataset Information")

st.write("First 10 Records")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ==========================================
# 16. Basic Statistical Information
# ==========================================

st.subheader("Basic Dataset Statistics")

st.write("Total Number of Images:", len(df))

st.write(
    "Number of Categories:",
    df["Category"].nunique()
)

st.write(
    "Most Common Category:",
    df["Category"].value_counts().idxmax()
)


# ==========================================
# 17. Project Conclusion
# ==========================================

st.divider()

st.header("📝 Project Conclusion")

st.write(
    "This project explores Tom and Jerry image categories "
    "using Exploratory Data Analysis."
)

st.write(
    "We used Pandas, NumPy, Matplotlib, Seaborn "
    "and Plotly to analyze and visualize the dataset."
)

st.success(
    "Tom and Jerry EDA Dashboard is Running Successfully!"
)