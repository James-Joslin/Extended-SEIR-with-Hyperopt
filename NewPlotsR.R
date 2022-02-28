library(gam)
library(forcats)
library(mgcv)
library(reshape2)
library(ggpubr)
library(tidyverse)
as_datetime <- function(x) as.POSIXct("2022-02-10") + as.difftime(x, units = "days") #nolint

# Baseline - Live
files <- list.files("./Forecasts_Base/", pattern = ".csv")
print(files)
df <- matrix(nrow = 0, ncol = 9)
df <- data.frame(df)
for (i in seq(1, length(files))) {
    in_file <- read.csv(paste("./Forecasts_Base/", files[i], sep = ""))
    df <- rbind(df, in_file)
}
df <- df[c("T", "F", "H", "Iasym", "Isym", "R")]
df <- df[order(df$T), ]
colnames(df) <- c("T", "F", "Live_H", "Live_Iasym", "Live_Isym", "R")
df$I = df$Live_Iasym + df$Live_Isym
dates <- df$T
dates <- as.data.frame(as_datetime(dates))
colnames(dates) <- c("Date")
df <- cbind(df, dates)
df <- df[c("Date", "I", "Live_H", "F")]
df$Category <- "Persistent Immunity"
head(df)


# waning 50+ Live
files <- list.files("./Forecasts_High/", pattern = ".csv")
df_high <- matrix(nrow = 0, ncol = 9)
df_high <- data.frame(df_high)
for (i in seq(1, length(files))) {
    in_file <- read.csv(paste("./Forecasts_High/", files[i], sep = ""))
    df_high <- rbind(df_high, in_file)
}
df_high <- df_high[c("T", "F", "H", "Iasym", "Isym", "R")]
df_high <- df_high[order(df_high$T), ]
colnames(df_high) <- c("T", "F", "Live_H", "Live_Iasym", "Live_Isym", "R")
df_high$I = df_high$Live_Iasym + df_high$Live_Isym
dates <- df_high$T
head(dates)
dates <- as.data.frame(as_datetime(dates))
colnames(dates) <- c("Date")
df_high <- cbind(df_high, dates)
df_high <- df_high[c("Date", "I", "Live_H", "F")]
df_high$Category <- "Waning Immunity - 50+"
head(df_high)


# waning all Live
files <- list.files("./Forecasts_ExtremeHigh/", pattern = ".csv")
df_ExHigh <- matrix(nrow = 0, ncol = 9)
df_ExHigh <- data.frame(df_ExHigh)
for (i in seq(1, length(files))) {
    in_file <- read.csv(paste("./Forecasts_High/", files[i], sep = ""))
    df_ExHigh <- rbind(df_ExHigh, in_file)
}
df_ExHigh <- df_ExHigh[c("T", "F", "H", "Iasym", "Isym", "R")]
df_ExHigh <- df_ExHigh[order(df_ExHigh$T), ]
colnames(df_ExHigh) <- c("T", "F", "Live_H", "Live_Iasym", "Live_Isym", "R")
df_ExHigh$I = df_ExHigh$Live_Iasym + df_ExHigh$Live_Isym
dates <- df_ExHigh$T
head(dates)
dates <- as.data.frame(as_datetime(dates))
colnames(dates) <- c("Date")
df_ExHigh <- cbind(df_ExHigh, dates)
df_ExHigh <- df_ExHigh[c("Date", "I", "Live_H", "F")]
df_ExHigh$Category <- "Waning Immunity - All age groups"
head(df_ExHigh)


# # Extreme High 2 - Live
# files <- list.files("./Forecasts_ExtremeHigh2/", pattern = ".csv")
# df_ExHigh2 <- matrix(nrow = 0, ncol = 9)
# df_ExHigh2 <- data.frame(df_ExHigh2)
# for (i in seq(1, length(files))) {
#     in_file <- read.csv(paste("./Forecasts_High/", files[i], sep = ""))
#     df_ExHigh2 <- rbind(df_ExHigh2, in_file)
# }
# df_ExHigh2 <- df_ExHigh2[c("T", "F", "H", "Iasym", "Isym", "R")]
# df_ExHigh2 <- df_ExHigh2[order(df_ExHigh2$T), ]
# colnames(df_ExHigh2) <- c("T", "F", "Live_H", "Live_Iasym", "Live_Isym", "R")
# df_ExHigh2$I = df_ExHigh2$Live_Iasym + df_ExHigh2$Live_Isym
# dates <- df_ExHigh2$T
# head(dates)
# dates <- as.data.frame(as_datetime(dates))
# colnames(dates) <- c("Date")
# df_ExHigh2 <- cbind(df_ExHigh2, dates)
# df_ExHigh2 <- df_ExHigh2[c("Date", "F", "Live_H", "I")]
# df_ExHigh2$Category <- "Severe Waning Immunity"

Live_df = rbind(df, df_high, df_ExHigh)
Live_dfMelt <- melt(Live_df, id.vars = c("Date", "Category"))
Live_dfMelt$HighCI <- Live_dfMelt$value * 1.15
Live_dfMelt$LowCI <- Live_dfMelt$value * 0.85
head(Live_dfMelt)

facet_names <- c(
                    `F` = "Cumulative Deaths",
                    `Live_H` = "Live Hospitalisations",
                    `I` = "Live Infections"
                    )
tibble(Live_dfMelt)
Live_Outputs <- ggplot(data = Live_dfMelt) +
    geom_line(
            data = Live_dfMelt, aes(x = Date, y = value, colour = Category), stat="smooth",
            method = mgcv::gam,
            formula  = y ~ s(x, bs = "cs", k = 15), size = 2,
            se = FALSE, , linetype = "solid", alpha = 1) +
    geom_line(
            data = Live_dfMelt, aes(x = Date, y = HighCI, colour = Category), stat="smooth",
            method = mgcv::gam,
            formula  = y ~ s(x, bs = "cs", k = 15), size = .5,
            se = FALSE, , linetype = "dashed", alpha = 1) +
    geom_line(
            data = Live_dfMelt, aes(x = Date, y = LowCI, colour = Category), stat="smooth",
            method = mgcv::gam,
            formula  = y ~ s(x, bs = "cs", k = 15), size = .5,
            se = FALSE, , linetype = "dashed", alpha = 1) +
    facet_wrap(~Live_dfMelt$variable, scales = "free_y", strip.position = "left", labeller = as_labeller(facet_names)) +
    # theme_light() +
    theme(strip.background = element_blank(),
        strip.text = element_text(size = 22, face = "bold"),
        strip.placement = "outside",
        legend.position = "top",
        legend.title = element_text(size = 22, face = "bold"),
        legend.text = element_text(size = 20),
        axis.text = element_text(size=20),
        axis.title = element_text(size=22,face="bold"),
        axis.title.y = element_blank(),
        plot.caption = element_text(hjust = 0, size = 22, face = "bold")) +
    coord_cartesian(ylim = c(-25, NA)) + 
    scale_x_datetime(date_breaks = "2 month", date_labels =  "%b")  +
    labs(caption = "\nPrediction intervals expressed as dotted lines above and below immunity categories")
# Live_Outputs
ggsave(filename = "./Plot_Outputs/liveplots.png", plot = Live_Outputs, scale = 2, width = 12, height = 7)

# Per day
# Base
files <- list.files("./Forecasts_Base/", pattern = ".csv")
print(files)
df <- matrix(nrow = 0, ncol = 2)
df <- data.frame(df)
for (i in seq(1, length(files))) {
    in_file <- read.csv(paste("./Forecasts_Base/", files[i], sep = ""))
    in_file$temp_totals <- in_file$Iasym + in_file$Isym + in_file$Q_Iasym + in_file$Q_Isym + in_file$Q_R + in_file$F + in_file$R
    dates <- in_file$T
    dates <- as.data.frame(as_datetime(dates))
    colnames(dates) <- c("Date")
    in_file <- cbind(in_file, dates)
    in_file = in_file[c("Date", "temp_totals")]
    in_file = tibble(in_file)
    in_file = in_file %>%
        mutate(diff = temp_totals - lag(temp_totals, default = first(temp_totals)))
    in_file = in_file[c("Date", "diff")]
    in_file$Date <- as.Date(in_file$Date)
    in_file <- aggregate(in_file["diff"], by=in_file["Date"], sum)
    df <- rbind(df, in_file)
}
df$Category <- "Persistent Immunity"
df <- df[order(df$Date), ]
df$row_num <- seq.int(nrow(df)) 
df$High_CI <- df$diff + ((df$row_num / length(df)) * 1.5)
df$LowCI <- df$diff - ((df$row_num / length(df)) * 0.65 )
head(df)

# waning 50+
files <- list.files("./Forecasts_High/", pattern = ".csv")
print(files)
df_high <- matrix(nrow = 0, ncol = 2)
df_high <- data.frame(df_high)
for (i in seq(1, length(files))) {
    in_file <- read.csv(paste("./Forecasts_High/", files[i], sep = ""))
    in_file$temp_totals <- in_file$Iasym + in_file$Isym + in_file$Q_Iasym + in_file$Q_Isym + in_file$Q_R + in_file$F + in_file$R
    dates <- in_file$T
    dates <- as.data.frame(as_datetime(dates))
    colnames(dates) <- c("Date")
    in_file <- cbind(in_file, dates)
    in_file = in_file[c("Date", "temp_totals")]
    in_file = tibble(in_file)
    in_file = in_file %>%
        mutate(diff = temp_totals - lag(temp_totals, default = first(temp_totals)))
    in_file = in_file[c("Date", "diff")]
    in_file$Date <- as.Date(in_file$Date)
    in_file <- aggregate(in_file["diff"], by=in_file["Date"], sum)
    df_high <- rbind(df_high, in_file)
}
df_high$Category <- "Waning Immunity - 50+"
df_high <- df_high[order(df_high$Date), ]
df_high$row_num <- seq.int(nrow(df_high)) 
df_high$High_CI <- df_high$diff + ((df_high$row_num / length(df_high)) * 1.5)
df_high$LowCI <- df_high$diff - ((df_high$row_num / length(df_high)) * 0.65 )
head(df_high)

# waning all
files <- list.files("./Forecasts_ExtremeHigh2/", pattern = ".csv")
print(files)
df_ExHigh <- matrix(nrow = 0, ncol = 2)
df_ExHigh <- data.frame(df_ExHigh)
for (i in seq(1, length(files))) {
    in_file <- read.csv(paste("./Forecasts_High/", files[i], sep = ""))
    in_file$temp_totals <- in_file$Iasym + in_file$Isym + in_file$Q_Iasym + in_file$Q_Isym + in_file$Q_R + in_file$F + in_file$R
    dates <- in_file$T
    dates <- as.data.frame(as_datetime(dates))
    colnames(dates) <- c("Date")
    in_file <- cbind(in_file, dates)
    in_file = in_file[c("Date", "temp_totals")]
    in_file = tibble(in_file)
    in_file = in_file %>%
        mutate(diff = temp_totals - lag(temp_totals, default = first(temp_totals)))
    in_file = in_file[c("Date", "diff")]
    in_file$Date <- as.Date(in_file$Date)
    in_file <- aggregate(in_file["diff"], by=in_file["Date"], sum)
    df_ExHigh <- rbind(df_ExHigh, in_file)
}
df_ExHigh$Category <- "Waning Immunity - All Age Groups"
df_ExHigh <- df_ExHigh[order(df_ExHigh$Date), ]
df_ExHigh$row_num <- seq.int(nrow(df_ExHigh)) 
df_ExHigh$High_CI <- df_ExHigh$diff + ((df_ExHigh$row_num / length(df_ExHigh)) * 1.5)
df_ExHigh$LowCI <- df_ExHigh$diff - ((df_ExHigh$row_num / length(df_ExHigh)) * 0.65 )
head(df_ExHigh)

df_perday <- rbind(df, df_high, df_ExHigh)
LowerCI_Vec <- df_perday$LowCI
LowerCI_Vec[LowerCI_Vec < 0] <- 0
df_perday$LowCI <- LowerCI_Vec
head(df_perday)

perDay_plot <- ggplot() +
    geom_line(
            data = df_perday, aes(x = Date, y = diff, color = Category), stat="smooth",
            method = mgcv::gam,
            formula  = y ~ s(x, bs = "cs", k = 15), size = 2,
            se = FALSE, , linetype = "solid", alpha = 1) +
    geom_line(
            data = df_perday, aes(x = Date, y = High_CI, color = Category), stat="smooth",
            method = mgcv::gam,
            formula  = y ~ s(x, bs = "cs", k = 15), size = 1,
            se = FALSE, , linetype = "dashed", alpha = .5) +
    geom_line(
            data = df_perday, aes(x = Date, y = LowCI, color = Category), stat="smooth",
            method = mgcv::gam,
            formula  = y ~ s(x, bs = "cs", k = 15), size = 1,
            se = FALSE, , linetype = "dashed", alpha = .5) +
    ylab("Cases Per Day") +
    scale_linetype_manual(values = c("Prediction Intervals")) +
    # theme_light() +
    theme(strip.background = element_blank(),
        strip.text = element_text(size = 22, face = "bold"),
        strip.placement = "outside",
        legend.position = "top",
        legend.title = element_text(size = 22, face = "bold"),
        legend.text = element_text(size = 20),
        axis.text = element_text(size=20),
        axis.title = element_text(size=22,face="bold"),
        plot.caption = element_text(hjust = 0, size = 22, face = "bold")
        ) +
    coord_cartesian(ylim = c(0, NA)) +
    labs(caption = "\nPrediction intervals expressed as dotted lines above and below immunity categories")
ggsave(filename = "./Plot_Outputs/perdayplot.png", plot = perDay_plot, scale = 2, width = 12, height = 7)
