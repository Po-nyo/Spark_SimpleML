import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.SparkSession

object Simple_ML {

  def main(args: Array[String]): Unit = {
    // Spark Session 생성
    val spark = SparkSession.builder().appName("Simple ML").master("local").getOrCreate()

    // Data Load
    val path = "./data/simple-ml"
    var df = spark.read.json(path)
    df.orderBy("value2").show()

    /*
    // 데이터 셋 변환
    val supervised = new RFormula().setFormula("lab ~ . + color:value1 + color:value2")

    val fittedRF = supervised.fit(df)
    val preparedDF = fittedRF.transform(df)
    preparedDF.show()

    // 데이터 셋 분할
    val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))

    // Logistic regression
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    println(lr.explainParams())

    // fit
    val fittedLR = lr.fit(train)

    // predict
    fittedLR.transform(train).select("label", "prediction").show()
    */

    // pipeline
    val Array(train, test) = df.randomSplit(Array(0.7, 0.3))

    val rForm = new RFormula()
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")

    val stages = Array(rForm, lr)
    val pipeline = new Pipeline().setStages(stages)

    // model versions
    val params = new ParamGridBuilder()
      .addGrid(rForm.formula, Array(
      "lab ~ . + color:value1",
      "lab ~ . + color:value1 + color:value2"))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(lr.regParam, Array(0.1, 2.0))
      .build()

    // evaluate
    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("prediction")
      .setLabelCol("label")

    val tvs = new TrainValidationSplit()
      .setTrainRatio(0.75)
      .setEstimatorParamMaps(params)
      .setEstimator(pipeline)
      .setEvaluator(evaluator)

    val tvsFitted = tvs.fit(train)

    // result
    println(evaluator.evaluate(tvsFitted.transform(test)))  // 0.9347826086956521

    println("\nsummary")
    val trainedPipeline = tvsFitted.bestModel.asInstanceOf[PipelineModel]
    val trainedLR = trainedPipeline.stages(1).asInstanceOf[LogisticRegressionModel]
    val summaryLR = trainedLR.summary
    println(summaryLR.objectiveHistory.mkString("\n"))

    // save and load
    /*
    tvs.Fitted.write.overwrite().save(save_path)

    val model = TrainValidationSplitModel.load(model_path)
    model.transform(test)
    */

  }
}
