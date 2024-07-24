try:
    from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
    print("KerasRegressor module is available.")
except ImportError as e:
    print("Error importing KerasRegressor:", e)
