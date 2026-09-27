class TreeExplainerLike:
    def shap_values(self, X):
        self.expected_value = self.model.predict(X).mean()
        return self._values(X)
