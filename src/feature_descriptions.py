"""
Feature descriptions and natural-language interpretation for Home Credit features.
Provides plain-English definitions and contextual explanations for all 88 model features.
"""

FEATURE_METADATA = {
    # -------------------------------------------------------------
    # 1. Applicant Demographics & Personal Attributes
    # -------------------------------------------------------------
    "DAYS_BIRTH": {
        "label": "Applicant Age (Days / Years)",
        "category": "Demographics",
        "description": "Age of the applicant in days (recorded as a negative number relative to the application). Older borrowers statistically exhibit lower default probabilities due to stable income.",
    },
    "DAYS_EMPLOYED": {
        "label": "Employment Tenure (Days)",
        "category": "Employment",
        "description": "Days employed at current job. A longer positive tenure indicates job and income stability. Positive values (365243) represent unemployed/pensioners.",
    },
    "DAYS_REGISTRATION": {
        "label": "Registration Change Tenure",
        "category": "Demographics",
        "description": "Days before loan application that the applicant modified their official residence registration. Frequent address changes can indicate residential instability.",
    },
    "DAYS_ID_PUBLISH": {
        "label": "ID Document Age",
        "category": "Demographics",
        "description": "Days before loan application that the applicant renewed or published their identity document.",
    },
    "CODE_GENDER": {
        "label": "Gender",
        "category": "Demographics",
        "description": "Gender of the loan applicant as recorded in the identity document.",
    },
    "NAME_EDUCATION_TYPE": {
        "label": "Highest Education Level",
        "category": "Demographics",
        "description": "Highest level of formal education completed (e.g. Higher education, Secondary special). Higher educational attainment correlates with higher repayment capability.",
    },
    "NAME_FAMILY_STATUS": {
        "label": "Family / Marital Status",
        "category": "Demographics",
        "description": "Family and marital status (e.g. Married, Single, Civil marriage, Separated). Reflects household support structure and dependency obligations.",
    },
    "NAME_INCOME_TYPE": {
        "label": "Income Source Category",
        "category": "Employment",
        "description": "Primary source of applicant income (e.g. Working, Commercial associate, Pensioner, State servant). State servants and commercial associates often demonstrate reliable cash flows.",
    },
    "OCCUPATION_TYPE": {
        "label": "Professional Occupation",
        "category": "Employment",
        "description": "Job classification of the applicant (e.g. Managers, Laborers, Core staff, Accountants, Drivers). Professional roles carry lower default risk.",
    },
    "ORGANIZATION_TYPE": {
        "label": "Employing Organization Type",
        "category": "Employment",
        "description": "Type of enterprise where the applicant works (e.g. Government, Business Entity, Self-employed, Industry). Government and large enterprises indicate steady employment.",
    },
    "CNT_CHILDREN": {
        "label": "Number of Children",
        "category": "Demographics",
        "description": "Number of dependent children. Higher dependents increase household living expenses, potentially tightening discretionary cash flow.",
    },
    "CNT_FAM_MEMBERS": {
        "label": "Total Family Size",
        "category": "Demographics",
        "description": "Total number of family members residing in the applicant's household.",
    },
    "FLAG_OWN_CAR": {
        "label": "Vehicle Ownership Flag",
        "category": "Assets",
        "description": "Whether the applicant owns an automobile (1 = Yes, 0 = No). Vehicle ownership acts as a proxy for unencumbered tangible assets.",
    },
    "FLAG_OWN_REALTY": {
        "label": "Real Estate Ownership Flag",
        "category": "Assets",
        "description": "Whether the applicant owns a house or flat (1 = Yes, 0 = No). Real estate ownership provides strong collateral backing and credit stability.",
    },
    "OWN_CAR_AGE": {
        "label": "Vehicle Age (Years)",
        "category": "Assets",
        "description": "Age of the applicant's car in years. Owning a newer vehicle reflects asset liquidity and disposable wealth.",
    },

    # -------------------------------------------------------------
    # 2. Loan Terms & Core Financial Capacity
    # -------------------------------------------------------------
    "AMT_INCOME_TOTAL": {
        "label": "Total Annual Income",
        "category": "Financial Capacity",
        "description": "Verified annual income of the applicant. Higher income increases debt-service capacity and buffers against unexpected financial shocks.",
    },
    "AMT_CREDIT": {
        "label": "Requested Credit Amount",
        "category": "Loan Terms",
        "description": "Total loan principal amount requested by the borrower. Larger loan amounts carry inherently higher exposure and stricter debt servicing demands.",
    },
    "AMT_ANNUITY": {
        "label": "Monthly Loan Annuity (EMI)",
        "category": "Loan Terms",
        "description": "Required monthly loan installment payment. High annuities relative to cash flow restrict disposable income and increase default likelihood.",
    },
    "AMT_GOODS_PRICE": {
        "label": "Retail Price of Goods",
        "category": "Loan Terms",
        "description": "Total purchase price of consumer goods being financed. A loan amount close to or lower than the goods price indicates down payment participation.",
    },
    "NAME_CONTRACT_TYPE": {
        "label": "Contract Type",
        "category": "Loan Terms",
        "description": "Type of loan contract requested (Cash loan vs. Revolving consumer credit). Revolving credit typically carries higher interest charges.",
    },
    "credit_to_income": {
        "label": "Credit-to-Income Leverage Ratio",
        "category": "Financial Capacity",
        "description": "Total loan amount divided by annual income. Ratios above 3.0x - 4.0x represent heavy debt leverage, significantly increasing credit risk.",
    },
    "annuity_to_income": {
        "label": "Annuity-to-Income Debt Burden (DTI)",
        "category": "Financial Capacity",
        "description": "Monthly loan installment divided by income. Represents the proportion of income committed to servicing this loan.",
    },
    "annuity_to_credit": {
        "label": "Payment Speed / Interest Multiple",
        "category": "Loan Terms",
        "description": "Monthly installment divided by total loan amount. Higher ratios reflect shorter loan tenures or higher interest rates.",
    },

    # -------------------------------------------------------------
    # 3. External Credit Bureau Ratings
    # -------------------------------------------------------------
    "EXT_SOURCE_1": {
        "label": "External Credit Bureau Score 1",
        "category": "Credit Bureau",
        "description": "Normalized credit bureau score from external scoring agencies (e.g. CIBIL/Experian). Values range 0.0 to 1.0; higher scores represent superior creditworthiness.",
    },
    "EXT_SOURCE_2": {
        "label": "External Credit Bureau Score 2",
        "category": "Credit Bureau",
        "description": "Normalized credit bureau score from scoring agency 2. Statistically the strongest single predictor of loan repayment in the entire model.",
    },
    "EXT_SOURCE_3": {
        "label": "External Credit Bureau Score 3",
        "category": "Credit Bureau",
        "description": "Normalized credit bureau score from scoring agency 3. Higher values strongly correlate with successful loan completion.",
    },
    "ext_source_mean": {
        "label": "Average External Credit Bureau Score",
        "category": "Credit Bureau",
        "description": "Harmonized average across all available external bureau ratings. Scores above 0.50 indicate an established, prime borrower profile.",
    },

    # -------------------------------------------------------------
    # 4. External Credit History (Other Banks / Bureau)
    # -------------------------------------------------------------
    "bureau_count": {
        "label": "Total External Credit Lines",
        "category": "Credit Bureau",
        "description": "Total number of previous credit lines recorded at other financial institutions in the central credit bureau.",
    },
    "active_bureau_count": {
        "label": "Active External Loans",
        "category": "Credit Bureau",
        "description": "Number of currently ongoing, active loans at other banks. High active loan counts elevate monthly repayment strain.",
    },
    "closed_bureau_count": {
        "label": "Closed External Loans in Good Standing",
        "category": "Credit Bureau",
        "description": "Number of past loans at other lenders that were successfully paid off and closed. Proves historical credit reliability.",
    },
    "bureau_last_year_count": {
        "label": "New External Loans in Past Year",
        "category": "Credit Bureau",
        "description": "Number of new credit lines opened in the last 12 months. Multiple recent loans indicate aggressive credit seeking behavior.",
    },
    "total_bureau_credit": {
        "label": "Total External Credit Limit",
        "category": "Credit Bureau",
        "description": "Cumulative credit limit granted by other financial institutions across the applicant's lifetime.",
    },
    "total_bureau_debt": {
        "label": "Total External Debt Balance",
        "category": "Credit Bureau",
        "description": "Cumulative balance currently owed across all other financial institutions.",
    },
    "avg_bureau_credit": {
        "label": "Average External Credit Line Size",
        "category": "Credit Bureau",
        "description": "Average credit limit per loan account in the credit bureau.",
    },
    "avg_bureau_debt": {
        "label": "Average External Debt Per Loan",
        "category": "Credit Bureau",
        "description": "Average outstanding balance across all external loans.",
    },
    "bureau_debt_to_credit_ratio": {
        "label": "External Debt-to-Credit Utilization",
        "category": "Credit Bureau",
        "description": "Total external debt divided by total external credit. Ratios near or above 1.0 signal that the borrower has exhausted their borrowing capacity.",
    },
    "max_credit_day_overdue": {
        "label": "Worst External Past-Due Record (Days)",
        "category": "Credit Bureau",
        "description": "Maximum days past due recorded on any external loan. Any value greater than 30 or 90 days indicates prior default delinquency.",
    },
    "total_credit_day_overdue": {
        "label": "Cumulative External Days Overdue",
        "category": "Credit Bureau",
        "description": "Total aggregate overdue days across all external bureau accounts.",
    },
    "oldest_bureau_record_days": {
        "label": "Length of Credit History (Oldest Account)",
        "category": "Credit Bureau",
        "description": "Days since the applicant opened their very first loan account. A long credit history gives statistical confidence to the risk model.",
    },
    "latest_bureau_record_days": {
        "label": "Recency of Credit Activity",
        "category": "Credit Bureau",
        "description": "Days since the applicant opened their most recent loan. Recent credit activity confirms the borrower is active in the financial system.",
    },
    "avg_bureau_age": {
        "label": "Average Age of External Accounts",
        "category": "Credit Bureau",
        "description": "Average lifespan of accounts recorded in the credit bureau.",
    },
    "bureau_balance_record_count": {
        "label": "Monthly Bureau Monitoring Snapshots",
        "category": "Credit Bureau",
        "description": "Number of monthly status records tracking loan health in the bureau balance database.",
    },
    "avg_bureau_balance_month": {
        "label": "Average Bureau Monitoring Horizon",
        "category": "Credit Bureau",
        "description": "Average historical observation period (in months) for external credit health.",
    },
    "oldest_bureau_balance_month": {
        "label": "Deepest Bureau Tracking Horizon",
        "category": "Credit Bureau",
        "description": "Maximum number of months of credit history continuously reported to the credit bureau.",
    },
    "bureau_overdue_status_count": {
        "label": "Months with Overdue Status (Bureau)",
        "category": "Credit Bureau",
        "description": "Number of monthly snapshots where the borrower was officially reported as overdue by another bank.",
    },
    "bureau_closed_status_count": {
        "label": "Months with Closed / Repaid Status",
        "category": "Credit Bureau",
        "description": "Number of monthly snapshots confirming fully repaid and closed credit facilities.",
    },

    # -------------------------------------------------------------
    # 5. Prior Lending History with Home Credit
    # -------------------------------------------------------------
    "previous_application_count": {
        "label": "Total Prior Loan Applications",
        "category": "Lender History",
        "description": "Total number of previous loan applications submitted to this lender in the past.",
    },
    "previous_approved_count": {
        "label": "Prior Approved Loans",
        "category": "Lender History",
        "description": "Number of past loan applications successfully approved by this lender.",
    },
    "previous_refused_count": {
        "label": "Prior Rejected Applications",
        "category": "Lender History",
        "description": "Number of previous applications rejected by the lender. Multiple past refusals signal recurring policy violations or credit impairment.",
    },
    "previous_canceled_count": {
        "label": "Prior Canceled Applications",
        "category": "Lender History",
        "description": "Applications previously canceled by the client before loan agreement execution.",
    },
    "previous_apps_last_year": {
        "label": "Applications in Past Year",
        "category": "Lender History",
        "description": "Applications submitted within the last 12 months. Multiple applications indicate urgency or credit-shopping behavior.",
    },
    "avg_previous_application_amount": {
        "label": "Average Amount Requested in Past",
        "category": "Lender History",
        "description": "Average loan amount the applicant requested across previous applications.",
    },
    "avg_previous_credit_amount": {
        "label": "Average Amount Disbursed in Past",
        "category": "Lender History",
        "description": "Average loan amount actually approved and disbursed to the borrower in historical applications.",
    },
    "total_previous_application_amount": {
        "label": "Cumulative Requested Amount",
        "category": "Lender History",
        "description": "Total cumulative loan volume requested across all historical applications.",
    },
    "total_previous_credit_amount": {
        "label": "Cumulative Credit Borrowed in Past",
        "category": "Lender History",
        "description": "Total cumulative loan principal disbursed to the client across all past applications. A high figure reflects substantial prior borrowing experience.",
    },
    "avg_previous_annuity": {
        "label": "Average Historical Monthly Payment",
        "category": "Lender History",
        "description": "Average monthly payment obligation on past loans with the lender.",
    },
    "avg_previous_decision_days": {
        "label": "Average Recency of Past Decisions",
        "category": "Lender History",
        "description": "Average number of days elapsed since historical loan underwriting decisions were made.",
    },
    "oldest_previous_decision": {
        "label": "First Application Tenure (Days)",
        "category": "Lender History",
        "description": "Days since the applicant's very first loan interaction with the institution.",
    },
    "latest_previous_decision": {
        "label": "Most Recent Application (Days)",
        "category": "Lender History",
        "description": "Days since the applicant's most recent prior loan application.",
    },

    # -------------------------------------------------------------
    # 6. Point-of-Sale (POS) & Cash Loan Track Record
    # -------------------------------------------------------------
    "pos_record_count": {
        "label": "Total POS & Cash Loan Snapshots",
        "category": "Repayment Track Record",
        "description": "Number of monthly payment snapshots tracking past consumer electronics and point-of-sale loans.",
    },
    "max_pos_installments": {
        "label": "Maximum POS Loan Term (Months)",
        "category": "Repayment Track Record",
        "description": "Longest term (number of monthly installments) on prior point-of-sale loans. Long terms (e.g. 60 months) lock in ongoing monthly obligations for multiple years.",
    },
    "avg_pos_installments": {
        "label": "Average POS Loan Term (Months)",
        "category": "Repayment Track Record",
        "description": "Average repayment tenure in months across all prior consumer goods loans.",
    },
    "max_pos_dpd": {
        "label": "Maximum Days Past Due on POS Loans",
        "category": "Repayment Track Record",
        "description": "Worst recorded payment delay (in days) on prior point-of-sale loan installments.",
    },
    "avg_pos_dpd": {
        "label": "Average Days Past Due on POS Loans",
        "category": "Repayment Track Record",
        "description": "Average payment delay in days across all previous point-of-sale installments.",
    },
    "pos_dpd_count": {
        "label": "Count of Overdue POS Installments",
        "category": "Repayment Track Record",
        "description": "Number of monthly installments where the borrower missed the due date on POS loans.",
    },
    "max_pos_dpd_def": {
        "label": "Worst Default-Tolerance Past-Due (POS)",
        "category": "Repayment Track Record",
        "description": "Maximum days past due on accounts evaluated under strict default-tolerance thresholds.",
    },
    "avg_pos_dpd_def": {
        "label": "Average Default-Tolerance Past-Due (POS)",
        "category": "Repayment Track Record",
        "description": "Average days past due under default-tolerance rules on POS accounts.",
    },

    # -------------------------------------------------------------
    # 7. Installment Repayment Discipline
    # -------------------------------------------------------------
    "installment_count": {
        "label": "Total Historical Installments Paid",
        "category": "Repayment Discipline",
        "description": "Total individual loan installments paid over the borrower's history. A high installment count (e.g. >100) provides definitive empirical proof of disciplined debt servicing.",
    },
    "total_installment_amount": {
        "label": "Total Required Repayments",
        "category": "Repayment Discipline",
        "description": "Sum of all required monthly payments scheduled across the borrower's loan agreements.",
    },
    "total_payment_amount": {
        "label": "Total Actual Cash Paid",
        "category": "Repayment Discipline",
        "description": "Actual cumulative money paid by the borrower. Matching or exceeding total required payments proves zero default history.",
    },
    "avg_installment_amount": {
        "label": "Average Scheduled Installment",
        "category": "Repayment Discipline",
        "description": "Average monthly payment required across all historical loan agreements.",
    },
    "avg_payment_amount": {
        "label": "Average Actual Payment Made",
        "category": "Repayment Discipline",
        "description": "Average cash payment made per monthly cycle.",
    },
    "avg_payment_ratio": {
        "label": "Installment Payment Fulfillment Ratio",
        "category": "Repayment Discipline",
        "description": "Actual money paid divided by scheduled amount. A ratio of 1.00 indicates complete fulfillment; ratios under 1.00 indicate chronic underpayment.",
    },
    "underpayment_count": {
        "label": "Count of Underpaid Installments",
        "category": "Repayment Discipline",
        "description": "Number of instances where the applicant paid less than the full monthly amount due.",
    },
    "avg_payment_delay": {
        "label": "Average Payment Delay (Days)",
        "category": "Repayment Discipline",
        "description": "Average days between installment due date and actual payment date. Negative values indicate payments made ahead of time; positive values indicate habitual lateness.",
    },
    "max_payment_delay": {
        "label": "Maximum Payment Delay (Days)",
        "category": "Repayment Discipline",
        "description": "Worst payment delay (in days) across all loan installments. Delays exceeding 30 days trigger severe credit penalties.",
    },

    # -------------------------------------------------------------
    # 8. Credit Card Balance, Limits & Utilization
    # -------------------------------------------------------------
    "credit_card_record_count": {
        "label": "Credit Card Billing Statements",
        "category": "Credit Card Behavior",
        "description": "Number of monthly credit card billing statements on record.",
    },
    "avg_cc_balance": {
        "label": "Average Revolving Card Debt",
        "category": "Credit Card Behavior",
        "description": "Average monthly outstanding debt balance on credit cards.",
    },
    "max_cc_balance": {
        "label": "Peak Credit Card Balance",
        "category": "Credit Card Behavior",
        "description": "Highest credit card balance reached by the applicant across all recorded billing cycles.",
    },
    "avg_cc_limit": {
        "label": "Average Credit Card Limit",
        "category": "Credit Card Behavior",
        "description": "Average revolving credit limit granted across credit cards.",
    },
    "max_cc_limit": {
        "label": "Maximum Credit Card Limit",
        "category": "Credit Card Behavior",
        "description": "Highest revolving credit limit extended to the applicant.",
    },
    "avg_cc_utilization": {
        "label": "Average Credit Card Utilization Rate",
        "category": "Credit Card Behavior",
        "description": "Average balance divided by credit limit. Sustained utilization above 40-50% indicates ongoing reliance on revolving debt.",
    },
    "max_cc_utilization": {
        "label": "Peak Credit Card Utilization Rate",
        "category": "Credit Card Behavior",
        "description": "Highest credit card limit utilization observed. Utilization near or exceeding 90-100% signals acute credit distress.",
    },
    "total_cc_drawings": {
        "label": "Total Credit Card Cash / Purchase Draws",
        "category": "Credit Card Behavior",
        "description": "Total amount withdrawn or spent via revolving credit cards.",
    },
    "avg_cc_payment": {
        "label": "Average Monthly Card Repayment",
        "category": "Credit Card Behavior",
        "description": "Average monthly payment made towards revolving credit card debt.",
    },
    "max_cc_dpd": {
        "label": "Maximum Card Overdue Days",
        "category": "Credit Card Behavior",
        "description": "Worst payment delay (in days) recorded on revolving credit card bills.",
    },
    "avg_cc_dpd": {
        "label": "Average Card Overdue Days",
        "category": "Credit Card Behavior",
        "description": "Average days past due on credit card billing cycles.",
    },
    "cc_dpd_count": {
        "label": "Overdue Card Billing Cycles",
        "category": "Credit Card Behavior",
        "description": "Number of billing cycles where the minimum credit card payment was missed.",
    },
}


def get_feature_info(feature_name: str) -> dict:
    """Return friendly label, category, and plain English description for any feature."""
    if feature_name in FEATURE_METADATA:
        return FEATURE_METADATA[feature_name]
    
    clean_title = feature_name.replace("_", " ").title()
    return {
        "label": clean_title,
        "category": "Model Feature",
        "description": f"Statistical input feature ({clean_title}) utilized by the underwriting risk model.",
    }


def describe_feature(feature_name: str, value, shap_value: float) -> dict:
    """
    Produce structured explanation for a single applicant feature and its SHAP impact.
    """
    info = get_feature_info(feature_name)
    label = info["label"]
    desc = info["description"]

    # Natural-language effect explanation
    if shap_value > 0:
        effect = "increases default risk"
        explanation_phrase = f"{label} contributed positively to the risk score (+{shap_value:.4f}), making default more likely in the model's estimation."
    elif shap_value < 0:
        effect = "reduces default risk"
        explanation_phrase = f"{label} contributed negatively to the risk score ({shap_value:.4f}), supporting a lower probability of default."
    else:
        effect = "neutral"
        explanation_phrase = f"{label} had minimal statistical influence on the predicted outcome."

    return {
        "feature": feature_name,
        "label": label,
        "category": info.get("category", "General"),
        "value": value,
        "shap": shap_value,
        "effect": effect,
        "description": desc,
        "explanation": explanation_phrase,
    }