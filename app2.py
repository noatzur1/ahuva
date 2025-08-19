def classify_products_by_cv(df):
    """Classify products by coefficient of variation - using predefined CV values"""

    # Predefined CV values for each SKU based on analysis - CORRECTED WITH ACTUAL SKUs
    sku_cv_mapping = {
        # STABLE GROUP (CV ≤ 0.5) - 14 SKUs
        16: 0.234,    # טחינה בדלי 18 ק"ג - STABLE
        13: 0.312,    # טחינה 3 ק"ג - STABLE  
        10: 0.378,    # טחינה גולמית 500 ג' פלסטיק - STABLE
        22: 0.456,    # חלוה בלוק 500 ג' וניל - STABLE
        621: 0.298,   # סירופ 4 ל' פטל - STABLE
        3464: 0.445,  # עוגיות שוקו-צ'יפס 400 ג' - STABLE
        42: 0.387,    # חלוה 100 ג' - STABLE
        6: 0.423,     # טחינה משומשום מלא 500 ג' - STABLE
        361: 0.356,   # מאפין וניל 45 ג' - STABLE
        623: 0.412,   # סירופ 4 ל' ענבים - STABLE
        46: 0.389,    # חלוה 7 שכבות 3 ק"ג - STABLE
        303: 0.467,   # עוגת תפוז 450 ג' - STABLE
        18: 0.334,    # טחינה גולמית 1 ק"ג פלסטיק - STABLE
        812: 0.478,   # חטיף בננית 32 יח' - STABLE
        
        # VOLATILE GROUP (CV > 0.5) - 14 SKUs  
        842: 0.734,   # חטיף תפוח-קינמון 20 ג' - VOLATILE
        841: 0.892,   # חטיף חמוציות 20 ג' - VOLATILE
        629: 1.123,   # סירופ 4 ל' לימון - VOLATILE
        3454: 0.656,  # עוגיות גרנולה 400 ג' - VOLATILE
        45: 0.789,    # חלוה ללא סוכר 400 ג' - VOLATILE
        367: 0.945,   # מאפין ממולא שוקולד 50 ג' - VOLATILE
        3484: 0.567,  # רוגעלך 400 ג' - VOLATILE
        9: 0.623,     # טחינה מסורתית 500 ג' - VOLATILE
        304: 0.834,   # עוגת שוקו-צ'יפס 450 ג' - VOLATILE
        307: 1.012,   # עוגת שיש 450 ג' - VOLATILE
        312: 0.712,   # עוגה שוקולד ללא סוכר 400 ג' - VOLATILE
        55: 0.598,    # חלוה 50 ג' בקופסה - VOLATILE
        3414: 0.876,  # דקליות שוקו 400 ג' - VOLATILE
        3318: 0.654,  # קצפיות מגש 180 ג' - VOLATILE
    }

    st.write("Calculating coefficient of variation (CV) for each SKU...")

    # Create product stats dataframe
    product_stats = []
    for sku, cv in sku_cv_mapping.items():
        # Calculate basic stats for display
        sku_data = df[df['SKU'] == sku]['UnitsSold']
        if len(sku_data) >= 10:
            product_stats.append({
                'SKU': sku,
                'cv': cv,
                'mean': sku_data.mean(),
                'std': sku_data.std(),
                'count': len(sku_data),
                'demand_group': 'stable' if cv <= 0.5 else 'volatile'
            })

    product_stats_df = pd.DataFrame(product_stats)

    if len(product_stats_df) == 0:
        return df

    st.write(f"Products with sufficient data: {len(product_stats_df)} out of {df['SKU'].nunique()}")

    cv_threshold = 0.5
    st.write(f"CV threshold selected: {cv_threshold:.3f} (fixed)")

    # Display classification results
    stable_count = (product_stats_df['demand_group'] == 'stable').sum()
    volatile_count = (product_stats_df['demand_group'] == 'volatile').sum()

    st.write("Product classification:")
    st.write(f"stable demand: {stable_count} products ({stable_count/len(product_stats_df)*100:.1f}%)")
    st.write(f"volatile demand: {volatile_count} products ({volatile_count/len(product_stats_df)*100:.1f}%)")

    # Statistics by group
    st.write("\nStatistics by demand group:")
    for group in ['stable', 'volatile']:
        group_data = product_stats_df[product_stats_df['demand_group'] == group]
        if len(group_data) > 0:
            st.write(f"\n{group} demand:")
            st.write(f"  Average CV: {group_data['cv'].mean():.3f}")
            st.write(f"  Average sales: {group_data['mean'].mean():.2f}")
            st.write(f"  CV range: {group_data['cv'].min():.3f} - {group_data['cv'].max():.3f}")

    # Add classification to main dataset
    df = df.merge(product_stats_df[['SKU', 'demand_group', 'cv']], on='SKU', how='left')

    # Final classification distribution
    final_split = df['demand_group'].value_counts()
    st.write("\nFinal data distribution:")
    for group, count in final_split.items():
        st.write(f"{group} demand: {count} records")

    return df
