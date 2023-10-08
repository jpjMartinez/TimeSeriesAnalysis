library(forecast)
library(readxl)
library(here)
library(xts)
library(openxlsx)

library(stats)
library(ggplot2)
library(forecast)
library(dplyr)

# =========================================================================

# Function to plot ACF and PACF
acf_pacf_grid <- function(ts, ts_name, num_lags = 50) {
  # Calculate ACF and PACF
  acf_result <- Acf(ts, lag.max = num_lags, plot = FALSE)
  pacf_result <- Pacf(ts, lag.max = num_lags, plot = FALSE)
  
  # Create an empty 1x2 grid
  par(mfrow = c(1, 2))

  # Plot ACF
  plot(acf_result$lag, acf_result$acf, type = 'h', main = NULL, xlab = 'Lag', ylab = 'FAC', col = 'blue')
  
  # Plot PACF
  plot(pacf_result$lag, pacf_result$acf, type = 'h', main = NULL, xlab = 'Lag', ylab = 'FACP', col = 'blue')

  # Plot a title in the center of the grid
  title(main = paste('FAC / FACP -', ts_name), outer = TRUE, line = -2, cex.main = 1.2)
}


# Function to plot a line chart
plot_chart <- function(x, y, x_label, y_label, title, line_color = 'blue', line_size = 1.5, smoothness = 0.2) {
  # Create a line plot using ggplot2
  df <- data.frame(x = x, y = y)
  ggplot(df, aes(x = x, y = y)) +
    geom_line(color = line_color, size = line_size, alpha = smoothness) +
    labs(x = x_label, y = y_label, title = title) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}

# =========================================================================


# Define the Excel file name
file_name <- "treated_ger_energ_ele_hidr.xlsx"

# Define the relative path to the Excel file in the "data sets" folder
file_path <- "C:/Users/João Martinez/Documents/12º Período/ENG1469 - Análise de Séries Temporais/listas_praticas/TimeSeriesAnalysis/datasets/%s" 

# Read the Excel file into a Data Frame
df <- read_excel(sprintf(file_path, file_name))

# Number of observations to reserve for the test set (last 12)
test_size <- 12

# Create the training set (all rows except the last 12)
train_set <- df[1:(nrow(df) - test_size), ]

# Create the test set (last 12 rows)
test_set <- df[(nrow(df) - test_size + 1):nrow(df), ]

# Create a time series (ts) object using xts
ts <- xts(train_set$geracao_gwh, order.by = train_set$mes_ano)

# Calculate the first difference and store it in ts_diff_1
ts_diff_1 <- diff(ts)

# Calculate the second difference and store it in ts_diff_2
ts_diff_2 <- diff(ts_diff_1)

# --------------------------------------------------------------------

# Plot the original time series
plot_chart(
  x=index(ts), y=ts,
  title='Geração de energia elétrica hidráulica',
  x_label='Tempo (mensal)',
  y_label='Geração (GWh)',
  line_size = 0.5,
  smoothness = 1.1,
  line_color='#375fa1'
)

# Plot the 1ª difference of the original time series
plot_chart(
  x=index(ts_diff_1), y=ts_diff_1,
  title='1ª Diferença da série temporal',
  x_label='Tempo (mensal)',
  y_label='Geração (GWh)',
  line_color='#32ad7a'
)

# Plot the 2ª difference of the original time series
plot_chart(
  x=index(ts_diff_2), y=ts_diff_2,
  title='2ª Diferença da série temporal',
  x_label='Tempo (mensal)',
  y_label='Geração (GWh)',
  line_color='#c26f2b'
)

# --------------------------------------------------------------------

# Plot the ACF and PACF of the 2ª diff. of the original series
acf_pacf_grid(ts_diff_2, '2ª Diferença da série temporal')

# --------------------------------------------------------------------
# Initialize lists to store ARMA models and AICc values separately
arma_list <- list()
aic_list <- list()

# Loop through different combinations of p and q
for (p in 0:5) {
  for (q in 0:5) {
    tryCatch({
      # Fit an ARMA(p, q, d) model using Arima
      model <- Arima(ts_diff_2, order=c(p, 0, q), seasonal=c(0, 0, 0, 0))
      
      # Construct the ARMA(p,q) string and append it to the arma_list
      arma_list <- append(arma_list, paste0("(", p, ",", q, ")"))
      
      # Append the AICc value to the aic_list
      aic_list <- append(aic_list, AIC(model, k = 2))  # k = 2 for AICc
    }, error = function(e) {
      # Handle any errors during model fitting
      cat("Error:", e$message, "\n")
    })
  }
}


# Create a dataframe from the two lists
results_df <- data.frame(ARMA = unlist(arma_list), AICc = unlist(aic_list))

# Print the final dataframe
print(results_df)

# Find the row index with the minimum AICc value
min_aicc_index <- which.min(results_df$AICc)

# Extract the corresponding "ARMA" value
best_arma <- results_df$ARMA[min_aicc_index]

# Print the best ARMA model order
cat("Best ARMA Model (Minimum AICc):", best_arma, "\n")


# Define the file name for the Excel file (replace with your desired file path)
excel_file <- "arma_results.xlsx"

# Write the dataframe to an Excel file
# write.xlsx(results_df, file = excel_file, sheetName = "ARMA_Results", rowNames = FALSE)

# Print a message to confirm the export
# cat("Dataframe exported to", excel_file, "\n")

    





