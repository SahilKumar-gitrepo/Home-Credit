---
license: apache-2.0
configs:
- config_name: POS_CASH_balance
  data_files:
  - split: train
    path: POS_CASH_balance/train-*
- config_name: application_test
  data_files:
  - split: train
    path: application_test/train-*
- config_name: application_train_dated
  data_files:
  - split: train
    path: application_train_dated/train-*
- config_name: bureau
  data_files:
  - split: train
    path: bureau/train-*
- config_name: bureau_balance
  data_files:
  - split: train
    path: bureau_balance/train-*
- config_name: credit_card_balance
  data_files:
  - split: train
    path: credit_card_balance/train-*
- config_name: installments_payments
  data_files:
  - split: train
    path: installments_payments/train-*
- config_name: previous_application
  data_files:
  - split: train
    path: previous_application/train-*
- config_name: sample_submission
  data_files:
  - split: train
    path: sample_submission/train-*
dataset_info:
- config_name: POS_CASH_balance
  features:
  - name: SK_ID_PREV
    dtype: int64
  - name: SK_ID_CURR
    dtype: int64
  - name: MONTHS_BALANCE
    dtype: int64
  - name: CNT_INSTALMENT
    dtype: float64
  - name: CNT_INSTALMENT_FUTURE
    dtype: float64
  - name: NAME_CONTRACT_STATUS
    dtype: string
  - name: SK_DPD
    dtype: int64
  - name: SK_DPD_DEF
    dtype: int64
  splits:
  - name: train
    num_bytes: 664921478
    num_examples: 10001358
  download_size: 178797309
  dataset_size: 664921478
- config_name: application_test
  features:
  - name: SK_ID_CURR
    dtype: int64
  - name: NAME_CONTRACT_TYPE
    dtype: string
  - name: CODE_GENDER
    dtype: string
  - name: FLAG_OWN_CAR
    dtype: string
  - name: FLAG_OWN_REALTY
    dtype: string
  - name: CNT_CHILDREN
    dtype: int64
  - name: AMT_INCOME_TOTAL
    dtype: float64
  - name: AMT_CREDIT
    dtype: float64
  - name: AMT_ANNUITY
    dtype: float64
  - name: AMT_GOODS_PRICE
    dtype: float64
  - name: NAME_TYPE_SUITE
    dtype: string
  - name: NAME_INCOME_TYPE
    dtype: string
  - name: NAME_EDUCATION_TYPE
    dtype: string
  - name: NAME_FAMILY_STATUS
    dtype: string
  - name: NAME_HOUSING_TYPE
    dtype: string
  - name: REGION_POPULATION_RELATIVE
    dtype: float64
  - name: DAYS_BIRTH
    dtype: int64
  - name: DAYS_EMPLOYED
    dtype: int64
  - name: DAYS_REGISTRATION
    dtype: float64
  - name: DAYS_ID_PUBLISH
    dtype: int64
  - name: OWN_CAR_AGE
    dtype: float64
  - name: FLAG_MOBIL
    dtype: int64
  - name: FLAG_EMP_PHONE
    dtype: int64
  - name: FLAG_WORK_PHONE
    dtype: int64
  - name: FLAG_CONT_MOBILE
    dtype: int64
  - name: FLAG_PHONE
    dtype: int64
  - name: FLAG_EMAIL
    dtype: int64
  - name: OCCUPATION_TYPE
    dtype: string
  - name: CNT_FAM_MEMBERS
    dtype: float64
  - name: REGION_RATING_CLIENT
    dtype: int64
  - name: REGION_RATING_CLIENT_W_CITY
    dtype: int64
  - name: WEEKDAY_APPR_PROCESS_START
    dtype: string
  - name: HOUR_APPR_PROCESS_START
    dtype: int64
  - name: REG_REGION_NOT_LIVE_REGION
    dtype: int64
  - name: REG_REGION_NOT_WORK_REGION
    dtype: int64
  - name: LIVE_REGION_NOT_WORK_REGION
    dtype: int64
  - name: REG_CITY_NOT_LIVE_CITY
    dtype: int64
  - name: REG_CITY_NOT_WORK_CITY
    dtype: int64
  - name: LIVE_CITY_NOT_WORK_CITY
    dtype: int64
  - name: ORGANIZATION_TYPE
    dtype: string
  - name: EXT_SOURCE_1
    dtype: float64
  - name: EXT_SOURCE_2
    dtype: float64
  - name: EXT_SOURCE_3
    dtype: float64
  - name: APARTMENTS_AVG
    dtype: float64
  - name: BASEMENTAREA_AVG
    dtype: float64
  - name: YEARS_BEGINEXPLUATATION_AVG
    dtype: float64
  - name: YEARS_BUILD_AVG
    dtype: float64
  - name: COMMONAREA_AVG
    dtype: float64
  - name: ELEVATORS_AVG
    dtype: float64
  - name: ENTRANCES_AVG
    dtype: float64
  - name: FLOORSMAX_AVG
    dtype: float64
  - name: FLOORSMIN_AVG
    dtype: float64
  - name: LANDAREA_AVG
    dtype: float64
  - name: LIVINGAPARTMENTS_AVG
    dtype: float64
  - name: LIVINGAREA_AVG
    dtype: float64
  - name: NONLIVINGAPARTMENTS_AVG
    dtype: float64
  - name: NONLIVINGAREA_AVG
    dtype: float64
  - name: APARTMENTS_MODE
    dtype: float64
  - name: BASEMENTAREA_MODE
    dtype: float64
  - name: YEARS_BEGINEXPLUATATION_MODE
    dtype: float64
  - name: YEARS_BUILD_MODE
    dtype: float64
  - name: COMMONAREA_MODE
    dtype: float64
  - name: ELEVATORS_MODE
    dtype: float64
  - name: ENTRANCES_MODE
    dtype: float64
  - name: FLOORSMAX_MODE
    dtype: float64
  - name: FLOORSMIN_MODE
    dtype: float64
  - name: LANDAREA_MODE
    dtype: float64
  - name: LIVINGAPARTMENTS_MODE
    dtype: float64
  - name: LIVINGAREA_MODE
    dtype: float64
  - name: NONLIVINGAPARTMENTS_MODE
    dtype: float64
  - name: NONLIVINGAREA_MODE
    dtype: float64
  - name: APARTMENTS_MEDI
    dtype: float64
  - name: BASEMENTAREA_MEDI
    dtype: float64
  - name: YEARS_BEGINEXPLUATATION_MEDI
    dtype: float64
  - name: YEARS_BUILD_MEDI
    dtype: float64
  - name: COMMONAREA_MEDI
    dtype: float64
  - name: ELEVATORS_MEDI
    dtype: float64
  - name: ENTRANCES_MEDI
    dtype: float64
  - name: FLOORSMAX_MEDI
    dtype: float64
  - name: FLOORSMIN_MEDI
    dtype: float64
  - name: LANDAREA_MEDI
    dtype: float64
  - name: LIVINGAPARTMENTS_MEDI
    dtype: float64
  - name: LIVINGAREA_MEDI
    dtype: float64
  - name: NONLIVINGAPARTMENTS_MEDI
    dtype: float64
  - name: NONLIVINGAREA_MEDI
    dtype: float64
  - name: FONDKAPREMONT_MODE
    dtype: string
  - name: HOUSETYPE_MODE
    dtype: string
  - name: TOTALAREA_MODE
    dtype: float64
  - name: WALLSMATERIAL_MODE
    dtype: string
  - name: EMERGENCYSTATE_MODE
    dtype: string
  - name: OBS_30_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: DEF_30_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: OBS_60_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: DEF_60_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: DAYS_LAST_PHONE_CHANGE
    dtype: float64
  - name: FLAG_DOCUMENT_2
    dtype: int64
  - name: FLAG_DOCUMENT_3
    dtype: int64
  - name: FLAG_DOCUMENT_4
    dtype: int64
  - name: FLAG_DOCUMENT_5
    dtype: int64
  - name: FLAG_DOCUMENT_6
    dtype: int64
  - name: FLAG_DOCUMENT_7
    dtype: int64
  - name: FLAG_DOCUMENT_8
    dtype: int64
  - name: FLAG_DOCUMENT_9
    dtype: int64
  - name: FLAG_DOCUMENT_10
    dtype: int64
  - name: FLAG_DOCUMENT_11
    dtype: int64
  - name: FLAG_DOCUMENT_12
    dtype: int64
  - name: FLAG_DOCUMENT_13
    dtype: int64
  - name: FLAG_DOCUMENT_14
    dtype: int64
  - name: FLAG_DOCUMENT_15
    dtype: int64
  - name: FLAG_DOCUMENT_16
    dtype: int64
  - name: FLAG_DOCUMENT_17
    dtype: int64
  - name: FLAG_DOCUMENT_18
    dtype: int64
  - name: FLAG_DOCUMENT_19
    dtype: int64
  - name: FLAG_DOCUMENT_20
    dtype: int64
  - name: FLAG_DOCUMENT_21
    dtype: int64
  - name: AMT_REQ_CREDIT_BUREAU_HOUR
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_DAY
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_WEEK
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_MON
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_QRT
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_YEAR
    dtype: float64
  splits:
  - name: train
    num_bytes: 50881873
    num_examples: 48744
  download_size: 9177686
  dataset_size: 50881873
- config_name: application_train_dated
  features:
  - name: SK_ID_CURR
    dtype: int64
  - name: application_date
    dtype: string
  - name: TARGET
    dtype: int64
  - name: NAME_CONTRACT_TYPE
    dtype: string
  - name: CODE_GENDER
    dtype: string
  - name: FLAG_OWN_CAR
    dtype: string
  - name: FLAG_OWN_REALTY
    dtype: string
  - name: CNT_CHILDREN
    dtype: int64
  - name: AMT_INCOME_TOTAL
    dtype: float64
  - name: AMT_CREDIT
    dtype: float64
  - name: AMT_ANNUITY
    dtype: float64
  - name: AMT_GOODS_PRICE
    dtype: float64
  - name: NAME_TYPE_SUITE
    dtype: string
  - name: NAME_INCOME_TYPE
    dtype: string
  - name: NAME_EDUCATION_TYPE
    dtype: string
  - name: NAME_FAMILY_STATUS
    dtype: string
  - name: NAME_HOUSING_TYPE
    dtype: string
  - name: REGION_POPULATION_RELATIVE
    dtype: float64
  - name: DAYS_BIRTH
    dtype: int64
  - name: DAYS_EMPLOYED
    dtype: int64
  - name: DAYS_REGISTRATION
    dtype: float64
  - name: DAYS_ID_PUBLISH
    dtype: int64
  - name: OWN_CAR_AGE
    dtype: float64
  - name: FLAG_MOBIL
    dtype: int64
  - name: FLAG_EMP_PHONE
    dtype: int64
  - name: FLAG_WORK_PHONE
    dtype: int64
  - name: FLAG_CONT_MOBILE
    dtype: int64
  - name: FLAG_PHONE
    dtype: int64
  - name: FLAG_EMAIL
    dtype: int64
  - name: OCCUPATION_TYPE
    dtype: string
  - name: CNT_FAM_MEMBERS
    dtype: float64
  - name: REGION_RATING_CLIENT
    dtype: int64
  - name: REGION_RATING_CLIENT_W_CITY
    dtype: int64
  - name: WEEKDAY_APPR_PROCESS_START
    dtype: string
  - name: HOUR_APPR_PROCESS_START
    dtype: int64
  - name: REG_REGION_NOT_LIVE_REGION
    dtype: int64
  - name: REG_REGION_NOT_WORK_REGION
    dtype: int64
  - name: LIVE_REGION_NOT_WORK_REGION
    dtype: int64
  - name: REG_CITY_NOT_LIVE_CITY
    dtype: int64
  - name: REG_CITY_NOT_WORK_CITY
    dtype: int64
  - name: LIVE_CITY_NOT_WORK_CITY
    dtype: int64
  - name: ORGANIZATION_TYPE
    dtype: string
  - name: EXT_SOURCE_1
    dtype: float64
  - name: EXT_SOURCE_2
    dtype: float64
  - name: EXT_SOURCE_3
    dtype: float64
  - name: APARTMENTS_AVG
    dtype: float64
  - name: BASEMENTAREA_AVG
    dtype: float64
  - name: YEARS_BEGINEXPLUATATION_AVG
    dtype: float64
  - name: YEARS_BUILD_AVG
    dtype: float64
  - name: COMMONAREA_AVG
    dtype: float64
  - name: ELEVATORS_AVG
    dtype: float64
  - name: ENTRANCES_AVG
    dtype: float64
  - name: FLOORSMAX_AVG
    dtype: float64
  - name: FLOORSMIN_AVG
    dtype: float64
  - name: LANDAREA_AVG
    dtype: float64
  - name: LIVINGAPARTMENTS_AVG
    dtype: float64
  - name: LIVINGAREA_AVG
    dtype: float64
  - name: NONLIVINGAPARTMENTS_AVG
    dtype: float64
  - name: NONLIVINGAREA_AVG
    dtype: float64
  - name: APARTMENTS_MODE
    dtype: float64
  - name: BASEMENTAREA_MODE
    dtype: float64
  - name: YEARS_BEGINEXPLUATATION_MODE
    dtype: float64
  - name: YEARS_BUILD_MODE
    dtype: float64
  - name: COMMONAREA_MODE
    dtype: float64
  - name: ELEVATORS_MODE
    dtype: float64
  - name: ENTRANCES_MODE
    dtype: float64
  - name: FLOORSMAX_MODE
    dtype: float64
  - name: FLOORSMIN_MODE
    dtype: float64
  - name: LANDAREA_MODE
    dtype: float64
  - name: LIVINGAPARTMENTS_MODE
    dtype: float64
  - name: LIVINGAREA_MODE
    dtype: float64
  - name: NONLIVINGAPARTMENTS_MODE
    dtype: float64
  - name: NONLIVINGAREA_MODE
    dtype: float64
  - name: APARTMENTS_MEDI
    dtype: float64
  - name: BASEMENTAREA_MEDI
    dtype: float64
  - name: YEARS_BEGINEXPLUATATION_MEDI
    dtype: float64
  - name: YEARS_BUILD_MEDI
    dtype: float64
  - name: COMMONAREA_MEDI
    dtype: float64
  - name: ELEVATORS_MEDI
    dtype: float64
  - name: ENTRANCES_MEDI
    dtype: float64
  - name: FLOORSMAX_MEDI
    dtype: float64
  - name: FLOORSMIN_MEDI
    dtype: float64
  - name: LANDAREA_MEDI
    dtype: float64
  - name: LIVINGAPARTMENTS_MEDI
    dtype: float64
  - name: LIVINGAREA_MEDI
    dtype: float64
  - name: NONLIVINGAPARTMENTS_MEDI
    dtype: float64
  - name: NONLIVINGAREA_MEDI
    dtype: float64
  - name: FONDKAPREMONT_MODE
    dtype: string
  - name: HOUSETYPE_MODE
    dtype: string
  - name: TOTALAREA_MODE
    dtype: float64
  - name: WALLSMATERIAL_MODE
    dtype: string
  - name: EMERGENCYSTATE_MODE
    dtype: string
  - name: OBS_30_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: DEF_30_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: OBS_60_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: DEF_60_CNT_SOCIAL_CIRCLE
    dtype: float64
  - name: DAYS_LAST_PHONE_CHANGE
    dtype: float64
  - name: FLAG_DOCUMENT_2
    dtype: int64
  - name: FLAG_DOCUMENT_3
    dtype: int64
  - name: FLAG_DOCUMENT_4
    dtype: int64
  - name: FLAG_DOCUMENT_5
    dtype: int64
  - name: FLAG_DOCUMENT_6
    dtype: int64
  - name: FLAG_DOCUMENT_7
    dtype: int64
  - name: FLAG_DOCUMENT_8
    dtype: int64
  - name: FLAG_DOCUMENT_9
    dtype: int64
  - name: FLAG_DOCUMENT_10
    dtype: int64
  - name: FLAG_DOCUMENT_11
    dtype: int64
  - name: FLAG_DOCUMENT_12
    dtype: int64
  - name: FLAG_DOCUMENT_13
    dtype: int64
  - name: FLAG_DOCUMENT_14
    dtype: int64
  - name: FLAG_DOCUMENT_15
    dtype: int64
  - name: FLAG_DOCUMENT_16
    dtype: int64
  - name: FLAG_DOCUMENT_17
    dtype: int64
  - name: FLAG_DOCUMENT_18
    dtype: int64
  - name: FLAG_DOCUMENT_19
    dtype: int64
  - name: FLAG_DOCUMENT_20
    dtype: int64
  - name: FLAG_DOCUMENT_21
    dtype: int64
  - name: AMT_REQ_CREDIT_BUREAU_HOUR
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_DAY
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_WEEK
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_MON
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_QRT
    dtype: float64
  - name: AMT_REQ_CREDIT_BUREAU_YEAR
    dtype: float64
  splits:
  - name: train
    num_bytes: 327982315
    num_examples: 307511
  download_size: 57065624
  dataset_size: 327982315
- config_name: bureau
  features:
  - name: SK_ID_CURR
    dtype: int64
  - name: SK_ID_BUREAU
    dtype: int64
  - name: CREDIT_ACTIVE
    dtype: string
  - name: CREDIT_CURRENCY
    dtype: string
  - name: DAYS_CREDIT
    dtype: int64
  - name: CREDIT_DAY_OVERDUE
    dtype: int64
  - name: DAYS_CREDIT_ENDDATE
    dtype: float64
  - name: DAYS_ENDDATE_FACT
    dtype: float64
  - name: AMT_CREDIT_MAX_OVERDUE
    dtype: float64
  - name: CNT_CREDIT_PROLONG
    dtype: int64
  - name: AMT_CREDIT_SUM
    dtype: float64
  - name: AMT_CREDIT_SUM_DEBT
    dtype: float64
  - name: AMT_CREDIT_SUM_LIMIT
    dtype: float64
  - name: AMT_CREDIT_SUM_OVERDUE
    dtype: float64
  - name: CREDIT_TYPE
    dtype: string
  - name: DAYS_CREDIT_UPDATE
    dtype: int64
  - name: AMT_ANNUITY
    dtype: float64
  splits:
  - name: train
    num_bytes: 265576748
    num_examples: 1716428
  download_size: 66538031
  dataset_size: 265576748
- config_name: bureau_balance
  features:
  - name: SK_ID_BUREAU
    dtype: int64
  - name: MONTHS_BALANCE
    dtype: int64
  - name: STATUS
    dtype: string
  splits:
  - name: train
    num_bytes: 573298425
    num_examples: 27299925
  download_size: 59790614
  dataset_size: 573298425
- config_name: credit_card_balance
  features:
  - name: SK_ID_PREV
    dtype: int64
  - name: SK_ID_CURR
    dtype: int64
  - name: MONTHS_BALANCE
    dtype: int64
  - name: AMT_BALANCE
    dtype: float64
  - name: AMT_CREDIT_LIMIT_ACTUAL
    dtype: int64
  - name: AMT_DRAWINGS_ATM_CURRENT
    dtype: float64
  - name: AMT_DRAWINGS_CURRENT
    dtype: float64
  - name: AMT_DRAWINGS_OTHER_CURRENT
    dtype: float64
  - name: AMT_DRAWINGS_POS_CURRENT
    dtype: float64
  - name: AMT_INST_MIN_REGULARITY
    dtype: float64
  - name: AMT_PAYMENT_CURRENT
    dtype: float64
  - name: AMT_PAYMENT_TOTAL_CURRENT
    dtype: float64
  - name: AMT_RECEIVABLE_PRINCIPAL
    dtype: float64
  - name: AMT_RECIVABLE
    dtype: float64
  - name: AMT_TOTAL_RECEIVABLE
    dtype: float64
  - name: CNT_DRAWINGS_ATM_CURRENT
    dtype: float64
  - name: CNT_DRAWINGS_CURRENT
    dtype: int64
  - name: CNT_DRAWINGS_OTHER_CURRENT
    dtype: float64
  - name: CNT_DRAWINGS_POS_CURRENT
    dtype: float64
  - name: CNT_INSTALMENT_MATURE_CUM
    dtype: float64
  - name: NAME_CONTRACT_STATUS
    dtype: string
  - name: SK_DPD
    dtype: int64
  - name: SK_DPD_DEF
    dtype: int64
  splits:
  - name: train
    num_bytes: 719008755
    num_examples: 3840312
  download_size: 182720694
  dataset_size: 719008755
- config_name: installments_payments
  features:
  - name: SK_ID_PREV
    dtype: int64
  - name: SK_ID_CURR
    dtype: int64
  - name: NUM_INSTALMENT_VERSION
    dtype: float64
  - name: NUM_INSTALMENT_NUMBER
    dtype: int64
  - name: DAYS_INSTALMENT
    dtype: float64
  - name: DAYS_ENTRY_PAYMENT
    dtype: float64
  - name: AMT_INSTALMENT
    dtype: float64
  - name: AMT_PAYMENT
    dtype: float64
  splits:
  - name: train
    num_bytes: 874147016
    num_examples: 13605401
  download_size: 525478733
  dataset_size: 874147016
- config_name: previous_application
  features:
  - name: SK_ID_PREV
    dtype: int64
  - name: SK_ID_CURR
    dtype: int64
  - name: NAME_CONTRACT_TYPE
    dtype: string
  - name: AMT_ANNUITY
    dtype: float64
  - name: AMT_APPLICATION
    dtype: float64
  - name: AMT_CREDIT
    dtype: float64
  - name: AMT_DOWN_PAYMENT
    dtype: float64
  - name: AMT_GOODS_PRICE
    dtype: float64
  - name: WEEKDAY_APPR_PROCESS_START
    dtype: string
  - name: HOUR_APPR_PROCESS_START
    dtype: int64
  - name: FLAG_LAST_APPL_PER_CONTRACT
    dtype: string
  - name: NFLAG_LAST_APPL_IN_DAY
    dtype: int64
  - name: RATE_DOWN_PAYMENT
    dtype: float64
  - name: RATE_INTEREST_PRIMARY
    dtype: float64
  - name: RATE_INTEREST_PRIVILEGED
    dtype: float64
  - name: NAME_CASH_LOAN_PURPOSE
    dtype: string
  - name: NAME_CONTRACT_STATUS
    dtype: string
  - name: DAYS_DECISION
    dtype: int64
  - name: NAME_PAYMENT_TYPE
    dtype: string
  - name: CODE_REJECT_REASON
    dtype: string
  - name: NAME_TYPE_SUITE
    dtype: string
  - name: NAME_CLIENT_TYPE
    dtype: string
  - name: NAME_GOODS_CATEGORY
    dtype: string
  - name: NAME_PORTFOLIO
    dtype: string
  - name: NAME_PRODUCT_TYPE
    dtype: string
  - name: CHANNEL_TYPE
    dtype: string
  - name: SELLERPLACE_AREA
    dtype: int64
  - name: NAME_SELLER_INDUSTRY
    dtype: string
  - name: CNT_PAYMENT
    dtype: float64
  - name: NAME_YIELD_GROUP
    dtype: string
  - name: PRODUCT_COMBINATION
    dtype: string
  - name: DAYS_FIRST_DRAWING
    dtype: float64
  - name: DAYS_FIRST_DUE
    dtype: float64
  - name: DAYS_LAST_DUE_1ST_VERSION
    dtype: float64
  - name: DAYS_LAST_DUE
    dtype: float64
  - name: DAYS_TERMINATION
    dtype: float64
  - name: NFLAG_INSURED_ON_APPROVAL
    dtype: float64
  splits:
  - name: train
    num_bytes: 600474354
    num_examples: 1670214
  download_size: 118136408
  dataset_size: 600474354
- config_name: sample_submission
  features:
  - name: SK_ID_CURR
    dtype: int64
  - name: TARGET
    dtype: float64
  splits:
  - name: train
    num_bytes: 779904
    num_examples: 48744
  download_size: 289637
  dataset_size: 779904
---
