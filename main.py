from src.config import DATA_PATH, COLUMNS_MISSING_VALUES, COLUMNS_LOW_CRAMERS_V, COLUMNS_LOW_SPEARMAN, EDUCATION_MAP, COLUMNS_MAJOR_DETAILED, CATEGORICAL_COLUMNS, NON_CATEGORICAL_COLUMNS, TARGET_COLUMN, WEIGHT_COLUMN, BASE_MODEL, PARAMS_GRID
from src.data_import import import_data
from src.data_preparation import prepare_data
from src.feature_engineering import perform_feature_eng
from src.data_resampling import calculate_weights
from src.model_evaluation import find_best_threshold, evaluate_model
from src.model_selection import select_best_model, print_feature_importance


def main():
    # Data Import
    print("Importing Data...")
    df_train, df_test = import_data(DATA_PATH)

    # Feature Engineering
    print("\nEngineering the features...")
    df_train['income bracket'] = df_train['income bracket'].map({'50000+.': 1, '- 50000.': 0})
    df_test['income bracket'] = df_test['income bracket'].map({'50000+.': 1, '- 50000.': 0})
    df_train, df_test = perform_feature_eng(
        df_train,
        df_test,
        columns_missing_values=COLUMNS_MISSING_VALUES,
        columns_low_cramer_v=COLUMNS_LOW_CRAMERS_V,
        columns_low_spearman=COLUMNS_LOW_SPEARMAN,
        education_map=EDUCATION_MAP,
        columns_major_detailed=COLUMNS_MAJOR_DETAILED,
    )

    # Data Preparation (incl. categorical encoding, data scaling)
    print("\nPreparing Data...")
    df_train, df_val, df_test = prepare_data(
        df_train,
        df_test,
        categorical_columns=CATEGORICAL_COLUMNS,
        non_categorical_columns=NON_CATEGORICAL_COLUMNS,
        target_col=TARGET_COLUMN,
        weight_col=WEIGHT_COLUMN,
    )

    # Calculate weights for resampling and representativity
    print("\nResampling Data...")
    class_counts = df_train[TARGET_COLUMN].value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    X_train, y_train, weights_train, \
    X_val, y_val, weights_val, \
    X_test, y_test, weights_test = calculate_weights(
        df_train,
        df_val,
        df_test,
        target_column=TARGET_COLUMN,
        weight_column=WEIGHT_COLUMN,
        scale_pos_weight=scale_pos_weight,
    )

    # Select best model
    print("\nSelecting Best Model...")
    best_model = select_best_model(
        BASE_MODEL,
        PARAMS_GRID,
        X_train,
        y_train,
        weights_train,
        X_val,
        y_val,
        weights_val,
    )
    # Show feature importance
    try:
        print_feature_importance(best_model, X_train)
    except AttributeError:
        print("Feature importance not available for this model.")

    # Evaluate best model
    print("\nEvaluating Best Model...")
    best_threshold = find_best_threshold(best_model, X_val, y_val, weights_val)
    evaluate_model(best_model, best_threshold, X_val, y_val, weights_val, 'Val')
    evaluate_model(best_model, best_threshold, X_test, y_test, weights_test, 'Test')


if __name__ == '__main__':
    main()
