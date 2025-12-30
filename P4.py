import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# تنظیم استایل سازگار با نسخه‌های جدید
sns.set_theme(style="darkgrid", palette="husl")


class KarateTalentML:
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model = None
        self.label_encoder = None

    def generate_dataset(self):
        """تولید دیتاست واقع‌بینانه برای کاراته"""
        print("🔧 در حال تولید دیتاست...")

        age = np.random.normal(16, 4, self.n_samples)
        age = np.clip(age, 8, 30)

        experience = np.random.exponential(18, self.n_samples)
        experience = np.clip(experience, 0, 60)

        base_talent = np.random.normal(0.6, 0.2, self.n_samples)

        data = {
            'age': age,
            'experience_months': experience,
            'flexibility': np.clip(base_talent * 70 + np.random.normal(0, 10, self.n_samples), 30, 90),
            'strength': np.clip((age / 30) * 60 + base_talent * 30 + np.random.normal(0, 15, self.n_samples), 40, 100),
            'reaction_time': np.clip(0.35 - (experience / 100) * 0.15 + np.random.normal(0, 0.04, self.n_samples), 0.15, 0.5),
            'speed': np.clip(1.2 + base_talent * 0.5 + np.random.normal(0, 0.2, self.n_samples), 0.8, 2.0),
            'balance': np.clip(base_talent * 80 + np.random.normal(0, 12, self.n_samples), 50, 100),
            'coordination': np.clip(base_talent * 75 + (experience / 60) * 15 + np.random.normal(0, 10, self.n_samples), 50, 100),
            'focus': np.clip(60 + base_talent * 25 + np.random.normal(0, 10, self.n_samples), 40, 100),
            'motivation': np.clip(np.random.normal(75, 15, self.n_samples), 40, 100),
            'endurance': np.clip(base_talent * 70 + (age / 30) * 20 + np.random.normal(0, 12, self.n_samples), 40, 100),
            'technique': np.clip((experience / 60) * 40 + base_talent * 40 + np.random.normal(0, 15, self.n_samples), 20, 100),
            'learning_speed': np.clip(base_talent * 80 + np.random.normal(0, 15, self.n_samples), 40, 100),
        }

        df = pd.DataFrame(data)

        talent_score = (
            0.15 * (df['flexibility'] / 90) +
            0.12 * (df['strength'] / 100) +
            0.12 * (1 - df['reaction_time'] / 0.5) +
            0.10 * (df['speed'] / 2.0) +
            0.10 * (df['balance'] / 100) +
            0.10 * (df['coordination'] / 100) +
            0.08 * (df['technique'] / 100) +
            0.08 * (df['learning_speed'] / 100) +
            0.07 * (df['endurance'] / 100) +
            0.04 * (df['focus'] / 100) +
            0.04 * (df['motivation'] / 100)
        )
        talent_score += np.random.normal(0, 0.05, self.n_samples)

        conditions = [
            talent_score >= 0.7,
            (talent_score >= 0.5) & (talent_score < 0.7),
            talent_score < 0.5
        ]
        choices = ['مستعد', 'متوسط', 'نیازمند تمرین']
        df['talent_class'] = np.select(conditions, choices, default='نیازمند تمرین')
        df['talent_score'] = talent_score

        print(f"✅ دیتاست با {self.n_samples} نمونه تولید شد")
        print(f"📊 توزیع کلاس‌ها:\n{df['talent_class'].value_counts()}")

        return df

    def prepare_data(self, df):
        print("\n🔧 آماده‌سازی داده‌ها...")

        feature_columns = [
            'age', 'experience_months', 'flexibility', 'strength',
            'reaction_time', 'speed', 'balance', 'coordination',
            'focus', 'motivation', 'endurance', 'technique', 'learning_speed'
        ]

        self.X = df[feature_columns]
        self.y = df['talent_class']

        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"📈 تعداد نمونه آموزش: {len(self.X_train)}")
        print(f"📊 تعداد نمونه تست: {len(self.X_test)}")

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_models(self):
        print("\n🚀 آموزش مدل‌های یادگیری ماشین...")

        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        }

        results = {}
        for name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, "predict_proba") else np.zeros((len(y_pred), len(np.unique(self.y_train))))
            accuracy = accuracy_score(self.y_test, y_pred)
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            print(f"✅ {name}: دقت={accuracy:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        self.best_model = results[best_model_name]['model']
        print(f"\n🏆 بهترین مدل: {best_model_name} با دقت {results[best_model_name]['accuracy']:.3f}")
        return results

    def evaluate_models(self, results):
        print("\n📊 ارزیابی مدل‌ها...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            cm = confusion_matrix(self.y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            ax.set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        axes[-1].axis('off')
        plt.tight_layout()
        plt.show()

        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        y_pred_best = results[best_model_name]['y_pred']
        print(f"\n📋 Classification Report for {best_model_name}:")
        print(classification_report(self.y_test, y_pred_best, target_names=self.label_encoder.classes_))

    def feature_importance(self):
        print("\n🔍 تحلیل اهمیت ویژگی‌ها...")
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_names = self.X.columns
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()
            print("🎯 Top Features:")
            for i, row in fi_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")

    def predict_new_player(self, player_data):
        print("\n🤔 پیش‌بینی استعداد ورزشکار جدید...")
        player_df = pd.DataFrame([player_data])
        player_scaled = self.scaler.transform(player_df)
        prediction = self.best_model.predict(player_scaled)
        prediction_proba = self.best_model.predict_proba(player_scaled)
        talent_class = self.label_encoder.inverse_transform(prediction)[0]
        probabilities = prediction_proba[0]
        print(f"🎯 نتیجه: {talent_class}")
        print(f"📊 احتمالات:")
        for i, prob in enumerate(probabilities):
            class_name = self.label_encoder.classes_[i]
            print(f"   {class_name}: {prob:.2%}")
        return talent_class, probabilities

    def save_model(self, filename='karate_talent_model.pkl'):
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': list(self.X.columns)
        }
        joblib.dump(model_data, filename)
        print(f"💾 مدل در فایل '{filename}' ذخیره شد")

    def visualize_dataset(self, df):
        print("\n📊 نمایش آماری دیتاست...")
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        class_counts = df['talent_class'].value_counts()
        axes[0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                    colors=['#2ecc71', '#f39c12', '#e74c3c'])
        axes[0].set_title('توزیع سطوح استعداد')

        sns.histplot(df['age'], kde=True, ax=axes[1], color='skyblue').set(title='توزیع سنی ورزشکاران')
        sns.histplot(df['talent_score'], kde=True, ax=axes[2], color='lightgreen').set(title='توزیع امتیاز استعداد')

        corr_cols = ['flexibility', 'strength', 'balance', 'coordination', 'talent_score']
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', center=0, ax=axes[3]).set(title='ماتریس همبستگی')
        sns.scatterplot(x='age', y='talent_score', hue='talent_class', data=df, ax=axes[4], palette='Set2').set(title='رابطه سن و امتیاز استعداد')

        boxplot_data = pd.melt(df, id_vars=['talent_class'], value_vars=['flexibility', 'strength', 'balance'],
                               var_name='ویژگی', value_name='امتیاز')
        sns.boxplot(x='ویژگی', y='امتیاز', hue='talent_class', data=boxplot_data, ax=axes[5], palette='pastel').set(title='مقایسه ویژگی‌های کلیدی')
        axes[-1].axis('off')
        plt.tight_layout()
        plt.show()


def main():
    print("=" * 60)
    print("🥋 سیستم هوشمند استعدادسنجی کاراته با یادگیری ماشین")
    print("=" * 60)

    talent_system = KarateTalentML(n_samples=1000)
    df = talent_system.generate_dataset()
    talent_system.visualize_dataset(df)
    X_train, X_test, y_train, y_test = talent_system.prepare_data(df)
    results = talent_system.train_models()
    talent_system.evaluate_models(results)
    talent_system.feature_importance()

    sample_player = {
        'age': 15,
        'experience_months': 12,
        'flexibility': 75,
        'strength': 65,
        'reaction_time': 0.28,
        'speed': 1.4,
        'balance': 85,
        'coordination': 80,
        'focus': 75,
        'motivation': 90,
        'endurance': 70,
        'technique': 60,
        'learning_speed': 85
    }
    talent_system.predict_new_player(sample_player)
    talent_system.save_model()
    print("\n✨ سیستم استعدادسنجی با موفقیت پیاده‌سازی شد!")
    return talent_system, df


if __name__ == "__main__":
    system, dataset = main()
    print("\n📄 نمونه‌ای از دیتاست تولید شده:")
    print(dataset[['age', 'experience_months', 'talent_class', 'talent_score']].head(10))

