using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.ML;
using System.Collections.Generic;

[TestClass]
public class TableRecommendationTests
{
    [TestMethod]
    public void TestDataLoading()
    {
        var dataList = new List<TableAssignmentData>();
        Assert.IsTrue(dataList.Count == 0, "Список данных должен быть пустым изначально");
        var data = new TableAssignmentData { BookingId = "synth_123", Label = true, Covers = 4.0f };
        dataList.Add(data);
        Assert.IsTrue(dataList.Count == 1, "Список данных должен содержать один элемент");
    }

    [TestMethod]
    public void TestModelPrediction()
    {
        var mlContext = new MLContext(seed: 0);
        var dataList = new List<TableAssignmentData> { 
            new TableAssignmentData { 
                BookingId = "synth_test", 
                Label = true, 
                Covers = 4.0f, 
                Area = "Main", 
                ClassWeight = 7.0f 
            } 
        };
        var data = mlContext.Data.LoadFromEnumerable(dataList);
        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("AreaEncoded", "Area")
            .Append(mlContext.Transforms.NormalizeMinMax("CoversNormalized", "Covers"))
            .Append(mlContext.BinaryClassification.Trainers.FastTree("Label", "Features", exampleWeightColumnName: "ClassWeight"));
        var model = pipeline.Fit(data);
        Assert.IsNotNull(model, "Модель должна быть успешно обучена");
    }

    [TestMethod]
    public void TestSyntheticDataGeneration()
    {
        var dataList = Program.GenerateSyntheticData();
        Assert.IsTrue(dataList.Count > 0, "Синтетические данные должны быть сгенерированы");
        Assert.IsTrue(dataList.All(d => d.BookingId.StartsWith("synth_")), "Все BookingId должны начинаться с 'synth_'");
    }
}