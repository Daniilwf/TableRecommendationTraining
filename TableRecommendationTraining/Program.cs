using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

public class TableAssignmentData
{
    [LoadColumn(0)] public string BookingId { get; set; }
    [LoadColumn(1)] public int MembershipId { get; set; }
    [LoadColumn(2)] public float IsVip { get; set; }
    [LoadColumn(3)] public float HandicapAccessible { get; set; }
    [LoadColumn(4)] public float Vegetarian { get; set; }
    [LoadColumn(5)] public float Covers { get; set; }
    [LoadColumn(6)] public float RequestedBooth { get; set; }
    [LoadColumn(7)] public float RequestedHighChair { get; set; }
    [LoadColumn(8)] public float RequestedStroller { get; set; }
    [LoadColumn(9)] public string TableNumber { get; set; }
    [LoadColumn(10)] public float TableTopSize { get; set; }
    [LoadColumn(11)] public string Area { get; set; }
    [LoadColumn(12)] public bool Label { get; set; }
    [LoadColumn(13)] public float VisitHour { get; set; }
    [LoadColumn(14)] public float AvgStayMinutes { get; set; }
    [LoadColumn(15)] public float ClassWeight { get; set; }
    [LoadColumn(16)] public float VisitDayOfWeek { get; set; }
    [LoadColumn(17)] public float IsPeakHour { get; set; }
    [LoadColumn(18)] public float BookingFrequency { get; set; }
    [LoadColumn(19)] public float PrefersQuiet { get; set; }
    [LoadColumn(20)] public float IsFrequentCustomer { get; set; }
    [LoadColumn(21)] public float TableOccupancyRate { get; set; }
    [LoadColumn(22)] public float IsWeekend { get; set; }
    [LoadColumn(23)] public float SpecialRequestCount { get; set; }
    [LoadColumn(24)] public float BookingRecency { get; set; }
    [LoadColumn(25)] public float AverageCoversPerBooking { get; set; }
    [LoadColumn(26)] public float SeasonalTrend { get; set; }
}

public class TableRecommendationPrediction
{
    [ColumnName("PredictedLabel")] public bool IsGoodMatch { get; set; }
    public float Probability { get; set; }
}

public class ModelMonitoring
{
    public void LogModelMetrics(float accuracy, float auc, float f1Score)
    {
        using (var writer = new System.IO.StreamWriter("metrics_log.txt", true))
        {
            writer.WriteLine($"Дата: {DateTime.Now}, Точность: {accuracy:P2}, AUC: {auc:P2}, F1 Score: {f1Score:P2}");
        }
    }

    public void LogPredictionError(string bookingId, bool predictedLabel, bool actualLabel)
    {
        if (predictedLabel != actualLabel)
        {
            using (var writer = new System.IO.StreamWriter("prediction_errors.txt", true))
            {
                writer.WriteLine($"Дата: {DateTime.Now}, BookingId: {bookingId}, Предсказание: {predictedLabel}, Фактическое: {actualLabel}");
            }
        }
    }
}

public class MLflowLogger
{
    private readonly HttpClient client = new HttpClient();

    public async Task LogMetrics(float accuracy, float auc, float f1Score)
    {
        var content = new StringContent(
            $"{{\"run_id\": \"demo_run\", \"metrics\": [{{\"key\": \"accuracy\", \"value\": {accuracy}}}, {{\"key\": \"auc\", \"value\": {auc}}}, {{\"key\": \"f1_score\", \"value\": {f1Score}}}]}}",
            Encoding.UTF8,
            "application/json"
        );
        await client.PostAsync("http://localhost:5000/api/2.0/mlflow/runs/log-metric", content);
    }
}

public class Program
{
    [SuppressMessage("ReSharper.DPA", "DPA0003: Excessive memory allocations in LOH", MessageId = "type: System.Double[]; size: 64MB")]
    static async Task Main(string[] args)
    {
        var mlContext = new MLContext(seed: 0);
        var monitoring = new ModelMonitoring();
        var mlflowLogger = new MLflowLogger();

        try
        {
            Console.WriteLine("Генерация синтетических данных для демонстрации...");
            var dataList = GenerateSyntheticData();
            Console.WriteLine($"Сгенерировано {dataList.Count} строк для обучения.");

            var dataFromEnumerable = mlContext.Data.LoadFromEnumerable(dataList);
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataFromEnumerable, testFraction: 0.2);
            var trainData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "AreaEncoded", inputColumnName: "Area")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VisitHourEncoded", inputColumnName: "VisitHour"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VisitDayOfWeekEncoded", inputColumnName: "VisitDayOfWeek"))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "CoversNormalized", inputColumnName: "Covers"))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "TableTopSizeNormalized", inputColumnName: "TableTopSize"))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "AvgStayMinutesNormalized", inputColumnName: "AvgStayMinutes"))
                .Append(mlContext.Transforms.NormalizeLogMeanVariance(outputColumnName: "BookingFrequencyNormalized", inputColumnName: "BookingFrequency"))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "TableOccupancyRateNormalized", inputColumnName: "TableOccupancyRate"))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "BookingRecencyNormalized", inputColumnName: "BookingRecency"))
                .Append(mlContext.Transforms.NormalizeMinMax(outputColumnName: "AverageCoversPerBookingNormalized", inputColumnName: "AverageCoversPerBooking"))
                .Append(mlContext.Transforms.Concatenate("Features", "IsVip", "HandicapAccessible", "Vegetarian", "CoversNormalized", "RequestedBooth", "RequestedHighChair", "RequestedStroller", "TableTopSizeNormalized", "AreaEncoded", "VisitHourEncoded", "VisitDayOfWeekEncoded", "AvgStayMinutesNormalized", "IsPeakHour", "BookingFrequencyNormalized", "PrefersQuiet", "IsFrequentCustomer", "TableOccupancyRateNormalized", "IsWeekend", "SpecialRequestCount", "BookingRecencyNormalized", "AverageCoversPerBookingNormalized", "SeasonalTrend"))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features", exampleWeightColumnName: "ClassWeight", 
                    numberOfLeaves: 20, minimumExampleCountPerLeaf: 10, learningRate: 0.05, numberOfTrees: 200));

            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Label");
            Console.WriteLine($"Точность: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
            Console.WriteLine($"Точность положительного класса: {metrics.PositivePrecision:P2}");
            Console.WriteLine($"Полнота положительного класса: {metrics.PositiveRecall:P2}");

            var crossValidationResults = mlContext.BinaryClassification.CrossValidate(dataFromEnumerable, pipeline, numberOfFolds: 5, labelColumnName: "Label");
            Console.WriteLine($"Кросс-валидация Точность: {crossValidationResults.Average(a => a.Metrics.Accuracy):P2}");
            Console.WriteLine($"Кросс-валидация AUC: {crossValidationResults.Average(a => a.Metrics.AreaUnderRocCurve):P2}");
            Console.WriteLine($"Кросс-валидация F1 Score: {crossValidationResults.Average(a => a.Metrics.F1Score):P2}");

            mlContext.Model.Save(model, trainData.Schema, "TableRecommendationModel.zip");
            Console.WriteLine("Модель сохранена в TableRecommendationModel.zip");

            monitoring.LogModelMetrics((float)metrics.Accuracy, (float)metrics.AreaUnderRocCurve, (float)metrics.F1Score);
            await mlflowLogger.LogMetrics((float)metrics.Accuracy, (float)metrics.AreaUnderRocCurve, (float)metrics.F1Score);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Произошла ошибка: {ex.Message}");
            monitoring.LogPredictionError("unknown", false, false);
        }
    }

    public static List<TableAssignmentData> GenerateSyntheticData()
    {
        var random = new Random();
        var dataList = new List<TableAssignmentData>();
        string[] areas = { "Main", "Terrace", "Private" };
        string[] tables = { "T1", "T2", "T3", "T4", "T5" };

        for (int i = 0; i < 15000; i++)
        {
            bool isPositive = random.NextDouble() > 0.33;
            var data = new TableAssignmentData
            {
                BookingId = $"synth_{i}",
                MembershipId = HashInt(random.Next(1000, 10000)),
                IsVip = random.NextDouble() > 0.8 ? 1.0f : 0.0f,
                HandicapAccessible = random.NextDouble() > 0.9 ? 1.0f : 0.0f,
                Vegetarian = random.NextDouble() > 0.7 ? 1.0f : 0.0f,
                Covers = (float)(random.Next(1, 10) + random.NextDouble()),
                RequestedBooth = random.NextDouble() > 0.6 ? 1.0f : 0.0f,
                RequestedHighChair = random.NextDouble() > 0.8 ? 1.0f : 0.0f,
                RequestedStroller = random.NextDouble() > 0.9 ? 1.0f : 0.0f,
                TableNumber = tables[random.Next(tables.Length)],
                TableTopSize = random.Next(2, 12),
                Area = areas[random.Next(areas.Length)],
                Label = isPositive,
                VisitHour = random.Next(8, 23),
                AvgStayMinutes = (float)(random.Next(30, 180) + random.NextDouble() * 10),
                ClassWeight = isPositive ? 7.0f : 2.0f,
                VisitDayOfWeek = random.Next(1, 8),
                IsPeakHour = random.NextDouble() > 0.5 ? 1.0f : 0.0f,
                BookingFrequency = random.Next(1, 20),
                PrefersQuiet = random.NextDouble() > 0.7 ? 1.0f : 0.0f,
                IsFrequentCustomer = random.NextDouble() > 0.6 ? 1.0f : 0.0f,
                TableOccupancyRate = (float)(random.NextDouble() * 0.5 + 0.5),
                IsWeekend = random.NextDouble() > 0.7 ? 1.0f : 0.0f,
                SpecialRequestCount = random.Next(0, 4),
                BookingRecency = random.Next(1, 60),
                AverageCoversPerBooking = (float)(random.Next(1, 10) + random.NextDouble()),
                SeasonalTrend = random.NextDouble() > 0.8 ? 1.0f : 0.0f
            };
            dataList.Add(data);
        }
        return dataList;
    }

    private static string HashString(string input) => Convert.ToBase64String(System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(input)));
    private static int HashInt(int input) => Math.Abs(input.GetHashCode());
}