Here’s the definitive source for the dataset we’ve been talking about:

📌 SIGIR 2021 E-Commerce Data Challenge (Coveo Data Challenge Dataset)
This dataset was released as part of the 2021 SIGIR eCommerce Workshop Data Challenge hosted by Coveo.
It’s a session-based clickstream dataset with millions of browsing events (views, adds, purchases) and associated shopping sessions—precisely the kind of data you need for your project.
📂 Official Source / Repository

The dataset (and associated challenge materials) is hosted in this GitHub repository:

The strategic goal is to use clickstream-based purchase predictions to forecast short-term product demand, guiding inventory replenishment decisions to reduce stockouts and overstock, and improve operational efficiency and profitability.

📌 SIGIR eCOM 2021 Data Challenge Repository (dataset + utilities)

Inside you’ll find:

browsing_train.csv — raw session browsing events
search_train.csv — search interactions
sku_to_content.csv — product metadata
README and challenge documentation

To obtain the full data, you may need to agree to terms & conditions and possibly register (it’s free for research/educational use).

📌 What This Dataset Contains

According to the SIGIR eCom Challenge description:

Session-based clickstream with ~36M events, including product interactions such as views, adds, and purchases
Search interactions with clicked and non-clicked items
Catalog metadata (e.g., SKU identifiers, content information)
Anonymized user/session hashes and timestamps

This gives you a rich signal for customer behavior, including:

browsing patterns
add-to-cart signals
early engagement
which items are ultimately purchased

—all of which you can use to link behavior to demand, just as we outlined in your pipeline.

🧠 Why This Source Is Ideal
Large-scale: tens of millions of events, high cardinality of sessions & SKUs
Behavior-rich: includes clicks, views, carts, search queries
Temporal: full timestamped logs suitable for time-window aggregation
Realistic: data from an actual e-commerce setting, not synthetic
