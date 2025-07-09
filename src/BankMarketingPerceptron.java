import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import javax.imageio.ImageIO;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

/**
 * Rede Neural para prever se o cliente fará aplicação bancária
 * Baseado no dataset Bank Marketing
 */
public class BankMarketingPerceptron {
    
    private static final int INPUT_SIZE = 16; // 16 atributos de entrada
    private static final int OUTPUT_SIZE = 1; // 1 saída (yes/no)
    
    // Mapas para codificar variáveis categóricas
    private static Map<String, Integer> jobMap = new HashMap<>();
    private static Map<String, Integer> maritalMap = new HashMap<>();
    private static Map<String, Integer> educationMap = new HashMap<>();
    private static Map<String, Integer> contactMap = new HashMap<>();
    private static Map<String, Integer> monthMap = new HashMap<>();
    private static Map<String, Integer> poutcomeMap = new HashMap<>();
    
    static {
        // Inicializar mapas de codificação
        String[] jobs = {"admin.", "unknown", "unemployed", "management", "housemaid", 
                        "entrepreneur", "student", "blue-collar", "self-employed", 
                        "retired", "technician", "services"};
        for (int i = 0; i < jobs.length; i++) {
            jobMap.put(jobs[i], i);
        }
        
        String[] marital = {"married", "divorced", "single"};
        for (int i = 0; i < marital.length; i++) {
            maritalMap.put(marital[i], i);
        }
        
        String[] education = {"unknown", "secondary", "primary", "tertiary"};
        for (int i = 0; i < education.length; i++) {
            educationMap.put(education[i], i);
        }
        
        String[] contact = {"unknown", "telephone", "cellular"};
        for (int i = 0; i < contact.length; i++) {
            contactMap.put(contact[i], i);
        }
        
        String[] months = {"jan", "feb", "mar", "apr", "may", "jun", 
                          "jul", "aug", "sep", "oct", "nov", "dec"};
        for (int i = 0; i < months.length; i++) {
            monthMap.put(months[i], i);
        }
        
        String[] poutcome = {"unknown", "other", "failure", "success"};
        for (int i = 0; i < poutcome.length; i++) {
            poutcomeMap.put(poutcome[i], i);
        }
    }
    
    public static void main(String[] args) {
        try {
            System.out.println("=== SISTEMA DE PREDIÇÃO DE CAMPANHAS BANCÁRIAS ===");
            System.out.println("Carregando dados de treinamento...");
            
            // Carregar dados de treinamento
            DataSet trainingSet = loadDataSet("bank_assets/bank.csv");
            System.out.println("Dados de treinamento carregados: " + trainingSet.size() + " registros");
            
            // Criar e treinar a rede neural
            System.out.println("\nTreinando rede neural...");
            MultiLayerPerceptron network = createAndTrainNetwork(trainingSet);
            
            // Salvar a rede treinada
            network.save("bankPredictionNetwork.nnet");
            System.out.println("Rede neural salva como 'bankPredictionNetwork.nnet'");
            
            // Carregar dados de teste (dataset completo)
            System.out.println("\nCarregando dados de teste...");
            DataSet testSet = loadDataSet("bank_assets/bank-full.csv");
            System.out.println("Dados de teste carregados: " + testSet.size() + " registros");
            
            // Avaliar o desempenho
            System.out.println("\n=== AVALIAÇÃO DO MODELO ===");
            evaluateModel(network, testSet);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private static DataSet loadDataSet(String filename) throws IOException {
        DataSet dataSet = new DataSet(INPUT_SIZE, OUTPUT_SIZE);
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine(); // Pular cabeçalho
            
            while ((line = br.readLine()) != null) {
                String[] values = line.replace("\"", "").split(";");
                
                if (values.length >= 17) {
                    double[] inputs = new double[INPUT_SIZE];
                    double[] outputs = new double[OUTPUT_SIZE];
                    
                    // Processar atributos de entrada
                    inputs[0] = normalizeAge(Double.parseDouble(values[0])); // age
                    inputs[1] = normalizeJob(values[1]); // job
                    inputs[2] = normalizeMarital(values[2]); // marital
                    inputs[3] = normalizeEducation(values[3]); // education
                    inputs[4] = values[4].equals("yes") ? 1.0 : 0.0; // default
                    inputs[5] = normalizeBalance(Double.parseDouble(values[5])); // balance
                    inputs[6] = values[6].equals("yes") ? 1.0 : 0.0; // housing
                    inputs[7] = values[7].equals("yes") ? 1.0 : 0.0; // loan
                    inputs[8] = normalizeContact(values[8]); // contact
                    inputs[9] = normalizeDay(Double.parseDouble(values[9])); // day
                    inputs[10] = normalizeMonth(values[10]); // month
                    inputs[11] = normalizeDuration(Double.parseDouble(values[11])); // duration
                    inputs[12] = normalizeCampaign(Double.parseDouble(values[12])); // campaign
                    inputs[13] = normalizePdays(Double.parseDouble(values[13])); // pdays
                    inputs[14] = normalizePrevious(Double.parseDouble(values[14])); // previous
                    inputs[15] = normalizePoutcome(values[15]); // poutcome
                    
                    // Processar saída
                    outputs[0] = values[16].equals("yes") ? 1.0 : 0.0; // y
                    
                    dataSet.addRow(new DataSetRow(inputs, outputs));
                }
            }
        }
        
        return dataSet;
    }
    
    private static MultiLayerPerceptron createAndTrainNetwork(DataSet trainingSet) {
        // Criar rede neural com arquitetura otimizada
        MultiLayerPerceptron network = new MultiLayerPerceptron(
            TransferFunctionType.SIGMOID, 
            INPUT_SIZE, 32, 16, OUTPUT_SIZE
        );
        
        // Configurar algoritmo de aprendizado
        BackPropagation learningRule = new BackPropagation();
        learningRule.setMaxIterations(5000);
        learningRule.setMaxError(0.02);
        learningRule.setLearningRate(0.1);
        
        // Treinar a rede
        network.learn(trainingSet, learningRule);
        
        return network;
    }
    
    private static void evaluateModel(NeuralNetwork<?> network, DataSet testSet) {
        int truePositives = 0;  // Previu SIM e era SIM
        int falsePositives = 0; // Previu SIM e era NÃO
        int trueNegatives = 0;  // Previu NÃO e era NÃO
        int falseNegatives = 0; // Previu NÃO e era SIM
        
        for (DataSetRow row : testSet.getRows()) {
            network.setInput(row.getInput());
            network.calculate();
            
            double prediction = network.getOutput()[0];
            double actual = row.getDesiredOutput()[0];
            
            boolean predictedPositive = prediction > 0.5;
            boolean actualPositive = actual > 0.5;
            
            if (predictedPositive && actualPositive) {
                truePositives++;
            } else if (predictedPositive && !actualPositive) {
                falsePositives++;
            } else if (!predictedPositive && !actualPositive) {
                trueNegatives++;
            } else {
                falseNegatives++;
            }
        }
        
        // Calcular métricas
        double accuracy = (double)(truePositives + trueNegatives) / 
                         (truePositives + trueNegatives + falsePositives + falseNegatives);
        
        double precision = truePositives > 0 ? (double)truePositives / (truePositives + falsePositives) : 0;
        double recall = truePositives > 0 ? (double)truePositives / (truePositives + falseNegatives) : 0;
        double f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        
        // Exibir resultados
        System.out.println("MATRIZ DE CONFUSÃO:");
        System.out.println("                 Previsto");
        System.out.println("                NÃO    SIM");
        System.out.println("Real    NÃO   " + String.format("%5d", trueNegatives) + "  " + String.format("%5d", falsePositives));
        System.out.println("        SIM   " + String.format("%5d", falseNegatives) + "  " + String.format("%5d", truePositives));
        
        System.out.println("\nMÉTRICAS DE DESEMPENHO:");
        System.out.println("Acurácia:  " + String.format("%.4f", accuracy) + " (" + String.format("%.2f", accuracy * 100) + "%)");
        System.out.println("Precisão:  " + String.format("%.4f", precision) + " (" + String.format("%.2f", precision * 100) + "%)");
        System.out.println("Revocação: " + String.format("%.4f", recall) + " (" + String.format("%.2f", recall * 100) + "%)");
        System.out.println("F1 Score:  " + String.format("%.4f", f1Score) + " (" + String.format("%.2f", f1Score * 100) + "%)");
        
        System.out.println("\nINTERPRETAÇÃO PARA CAMPANHA DE MARKETING:");
        System.out.println("- Acurácia: " + String.format("%.2f", accuracy * 100) + "% dos clientes são classificados corretamente");
        System.out.println("- Precisão: " + String.format("%.2f", precision * 100) + "% dos clientes identificados como 'SIM' realmente farão aplicação");
        System.out.println("- Revocação: " + String.format("%.2f", recall * 100) + "% dos clientes que farão aplicação são identificados corretamente");
        System.out.println("- F1 Score: " + String.format("%.2f", f1Score * 100) + "% - métrica balanceada entre precisão e revocação");
        
        // Análise de custo-benefício
        int totalContacts = truePositives + falsePositives;
        double efficiency = totalContacts > 0 ? (double)truePositives / totalContacts : 0;
        System.out.println("\nANÁLISE DE EFICIÊNCIA DA CAMPANHA:");
        System.out.println("- Clientes contactados: " + totalContacts);
        System.out.println("- Clientes que farão aplicação: " + truePositives);
        System.out.println("- Eficiência da campanha: " + String.format("%.2f", efficiency * 100) + "%");
        System.out.println("- Economia: Com este modelo, você pode focar em " + totalContacts + " clientes");
        System.out.println("  em vez de contactar todos os " + testSet.size() + " clientes do dataset");
        
        // Gerar imagem de visualização
        generateVisualization(network, testSet, truePositives, falsePositives, trueNegatives, falseNegatives);
    }
    
    /**
     * Gera uma visualização gráfica dos resultados da predição bancária
     */
    private static void generateVisualization(NeuralNetwork<?> network, DataSet testSet, 
                                            int truePositives, int falsePositives, 
                                            int trueNegatives, int falseNegatives) {
        
        BufferedImage img = new BufferedImage(1200, 800, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = (Graphics2D) img.getGraphics();
        
        // Configurar fundo
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, 1200, 800);
        
        // Título
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 24));
        g2d.drawString("ANÁLISE DE PREDIÇÃO - CAMPANHA BANCÁRIA", 300, 30);
        
        // Desenhar matriz de confusão
        drawConfusionMatrix(g2d, truePositives, falsePositives, trueNegatives, falseNegatives);
        
        // Desenhar gráfico de distribuição de probabilidades
        drawProbabilityDistribution(g2d, network, testSet);
        
        // Desenhar métricas
        drawMetrics(g2d, truePositives, falsePositives, trueNegatives, falseNegatives);
        
        // Salvar imagem
        try {
            ImageIO.write(img, "PNG", new File("BankPredictionVisualization.png"));
            System.out.println("\n📊 Visualização salva como 'BankPredictionVisualization.png'");
        } catch (IOException e) {
            System.err.println("Erro ao salvar imagem: " + e.getMessage());
        }
        
        g2d.dispose();
    }
    
    /**
     * Desenha a matriz de confusão
     */
    private static void drawConfusionMatrix(Graphics2D g2d, int tp, int fp, int tn, int fn) {
        int startX = 50;
        int startY = 80;
        int cellSize = 120;
        
        g2d.setFont(new Font("Arial", Font.BOLD, 16));
        g2d.setColor(Color.BLACK);
        g2d.drawString("MATRIZ DE CONFUSÃO", startX, startY - 20);
        
        // Cabeçalhos
        g2d.setFont(new Font("Arial", Font.BOLD, 14));
        g2d.drawString("Previsto", startX + cellSize, startY - 5);
        g2d.drawString("NÃO", startX + cellSize/2, startY + 15);
        g2d.drawString("SIM", startX + cellSize + cellSize/2, startY + 15);
        
        g2d.drawString("Real", startX - 40, startY + cellSize/2);
        g2d.drawString("NÃO", startX - 30, startY + cellSize/2 + 20);
        g2d.drawString("SIM", startX - 30, startY + cellSize + cellSize/2 + 20);
        
        // Células da matriz
        // Verdadeiros Negativos (TN)
        g2d.setColor(new Color(144, 238, 144)); // Verde claro
        g2d.fillRect(startX, startY + 30, cellSize, cellSize);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(startX, startY + 30, cellSize, cellSize);
        g2d.setFont(new Font("Arial", Font.BOLD, 18));
        g2d.drawString(String.valueOf(tn), startX + cellSize/2 - 20, startY + 30 + cellSize/2 + 5);
        
        // Falsos Positivos (FP)
        g2d.setColor(new Color(255, 182, 193)); // Rosa claro
        g2d.fillRect(startX + cellSize, startY + 30, cellSize, cellSize);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(startX + cellSize, startY + 30, cellSize, cellSize);
        g2d.drawString(String.valueOf(fp), startX + cellSize + cellSize/2 - 20, startY + 30 + cellSize/2 + 5);
        
        // Falsos Negativos (FN)
        g2d.setColor(new Color(255, 182, 193)); // Rosa claro
        g2d.fillRect(startX, startY + 30 + cellSize, cellSize, cellSize);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(startX, startY + 30 + cellSize, cellSize, cellSize);
        g2d.drawString(String.valueOf(fn), startX + cellSize/2 - 20, startY + 30 + cellSize + cellSize/2 + 5);
        
        // Verdadeiros Positivos (TP)
        g2d.setColor(new Color(144, 238, 144)); // Verde claro
        g2d.fillRect(startX + cellSize, startY + 30 + cellSize, cellSize, cellSize);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(startX + cellSize, startY + 30 + cellSize, cellSize, cellSize);
        g2d.drawString(String.valueOf(tp), startX + cellSize + cellSize/2 - 20, startY + 30 + cellSize + cellSize/2 + 5);
    }
    
    /**
     * Desenha a distribuição de probabilidades
     */
    private static void drawProbabilityDistribution(Graphics2D g2d, NeuralNetwork<?> network, DataSet testSet) {
        int startX = 400;
        int startY = 100;
        int width = 700;
        int height = 300;
        
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 16));
        g2d.drawString("DISTRIBUIÇÃO DE PROBABILIDADES", startX, startY - 20);
        
        // Eixos
        g2d.drawLine(startX, startY + height, startX + width, startY + height); // Eixo X
        g2d.drawLine(startX, startY, startX, startY + height); // Eixo Y
        
        // Labels dos eixos
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        g2d.drawString("Probabilidade", startX + width/2 - 30, startY + height + 20);
        g2d.drawString("Frequência", startX - 60, startY + height/2);
        
        // Criar histograma
        int[] histogram = new int[20]; // 20 bins para probabilidades de 0 a 1
        int totalSamples = 0;
        
        for (DataSetRow row : testSet.getRows()) {
            if (totalSamples >= 1000) break; // Limitar para performance
            
            network.setInput(row.getInput());
            network.calculate();
            double probability = network.getOutput()[0];
            
            int bin = Math.min((int)(probability * 20), 19);
            histogram[bin]++;
            totalSamples++;
        }
        
        // Encontrar máximo para normalização
        int maxFreq = 0;
        for (int freq : histogram) {
            maxFreq = Math.max(maxFreq, freq);
        }
        
        // Desenhar barras do histograma
        int barWidth = width / 20;
        for (int i = 0; i < 20; i++) {
            int barHeight = maxFreq > 0 ? (int)((double)histogram[i] / maxFreq * height) : 0;
            
            // Cor baseada na região (vermelho para baixa prob, verde para alta prob)
            float ratio = (float)i / 19;
            g2d.setColor(new Color(1.0f - ratio, ratio, 0.0f, 0.7f));
            g2d.fillRect(startX + i * barWidth, startY + height - barHeight, barWidth - 1, barHeight);
            
            g2d.setColor(Color.BLACK);
            g2d.drawRect(startX + i * barWidth, startY + height - barHeight, barWidth - 1, barHeight);
        }
        
        // Adicionar marcas no eixo X
        g2d.setColor(Color.BLACK);
        for (int i = 0; i <= 10; i++) {
            int x = startX + i * (width / 10);
            g2d.drawLine(x, startY + height, x, startY + height + 5);
            g2d.drawString(String.format("%.1f", i / 10.0), x - 5, startY + height + 18);
        }
    }
    
    /**
     * Desenha as métricas de desempenho
     */
    private static void drawMetrics(Graphics2D g2d, int tp, int fp, int tn, int fn) {
        int startX = 50;
        int startY = 450;
        
        double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);
        double precision = tp > 0 ? (double)tp / (tp + fp) : 0;
        double recall = tp > 0 ? (double)tp / (tp + fn) : 0;
        double f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 18));
        g2d.drawString("MÉTRICAS DE DESEMPENHO", startX, startY);
        
        g2d.setFont(new Font("Arial", Font.BOLD, 14));
        int lineHeight = 25;
        int currentY = startY + 30;
        
        // Acurácia
        g2d.setColor(getColorForMetric(accuracy));
        g2d.drawString("Acurácia:  " + String.format("%.2f%%", accuracy * 100), startX, currentY);
        drawProgressBar(g2d, startX + 200, currentY - 10, accuracy, getColorForMetric(accuracy));
        
        // Precisão
        currentY += lineHeight;
        g2d.setColor(getColorForMetric(precision));
        g2d.drawString("Precisão:  " + String.format("%.2f%%", precision * 100), startX, currentY);
        drawProgressBar(g2d, startX + 200, currentY - 10, precision, getColorForMetric(precision));
        
        // Revocação
        currentY += lineHeight;
        g2d.setColor(getColorForMetric(recall));
        g2d.drawString("Revocação: " + String.format("%.2f%%", recall * 100), startX, currentY);
        drawProgressBar(g2d, startX + 200, currentY - 10, recall, getColorForMetric(recall));
        
        // F1 Score
        currentY += lineHeight;
        g2d.setColor(getColorForMetric(f1Score));
        g2d.drawString("F1 Score:  " + String.format("%.2f%%", f1Score * 100), startX, currentY);
        drawProgressBar(g2d, startX + 200, currentY - 10, f1Score, getColorForMetric(f1Score));
        
        // Análise de eficiência
        currentY += lineHeight * 2;
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 16));
        g2d.drawString("ANÁLISE DE EFICIÊNCIA", startX, currentY);
        
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        currentY += 20;
        g2d.drawString("• Clientes contactados: " + (tp + fp) + " (em vez de " + (tp + tn + fp + fn) + ")", startX, currentY);
        currentY += 15;
        g2d.drawString("• Taxa de conversão: " + String.format("%.2f%%", precision * 100), startX, currentY);
        currentY += 15;
        double savings = 1.0 - (double)(tp + fp) / (tp + tn + fp + fn);
        g2d.drawString("• Economia de contactos: " + String.format("%.2f%%", savings * 100), startX, currentY);
    }
    
    /**
     * Retorna cor baseada no valor da métrica
     */
    private static Color getColorForMetric(double value) {
        if (value >= 0.8) return new Color(0, 150, 0); // Verde escuro
        if (value >= 0.6) return new Color(255, 165, 0); // Laranja
        return new Color(200, 0, 0); // Vermelho
    }
    
    /**
     * Desenha uma barra de progresso
     */
    private static void drawProgressBar(Graphics2D g2d, int x, int y, double value, Color color) {
        int barWidth = 200;
        int barHeight = 15;
        
        // Fundo da barra
        g2d.setColor(Color.LIGHT_GRAY);
        g2d.fillRect(x, y, barWidth, barHeight);
        
        // Barra de progresso
        g2d.setColor(color);
        g2d.fillRect(x, y, (int)(value * barWidth), barHeight);
        
        // Contorno
        g2d.setColor(Color.BLACK);
        g2d.drawRect(x, y, barWidth, barHeight);
    }
    
    // Métodos de normalização
    private static double normalizeAge(double age) {
        return age / 100.0; // Normalizar idade para [0,1]
    }
    
    private static double normalizeJob(String job) {
        return jobMap.getOrDefault(job, 0) / 11.0; // 12 categorias
    }
    
    private static double normalizeMarital(String marital) {
        return maritalMap.getOrDefault(marital, 0) / 2.0; // 3 categorias
    }
    
    private static double normalizeEducation(String education) {
        return educationMap.getOrDefault(education, 0) / 3.0; // 4 categorias
    }
    
    private static double normalizeBalance(double balance) {
        return Math.tanh(balance / 10000.0); // Normalizar com tanh para lidar com valores extremos
    }
    
    private static double normalizeContact(String contact) {
        return contactMap.getOrDefault(contact, 0) / 2.0; // 3 categorias
    }
    
    private static double normalizeDay(double day) {
        return day / 31.0; // Normalizar dia do mês
    }
    
    private static double normalizeMonth(String month) {
        return monthMap.getOrDefault(month, 0) / 11.0; // 12 meses
    }
    
    private static double normalizeDuration(double duration) {
        return Math.tanh(duration / 1000.0); // Normalizar duração
    }
    
    private static double normalizeCampaign(double campaign) {
        return Math.tanh(campaign / 10.0); // Normalizar número de campanhas
    }
    
    private static double normalizePdays(double pdays) {
        return pdays == -1 ? 0 : Math.tanh(pdays / 365.0); // Normalizar dias
    }
    
    private static double normalizePrevious(double previous) {
        return Math.tanh(previous / 10.0); // Normalizar contatos anteriores
    }
    
    private static double normalizePoutcome(String poutcome) {
        return poutcomeMap.getOrDefault(poutcome, 0) / 3.0; // 4 categorias
    }
}
