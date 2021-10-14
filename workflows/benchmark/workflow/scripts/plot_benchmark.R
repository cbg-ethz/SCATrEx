suppressWarnings(library(tidyverse))
suppressWarnings(library(ggpubr))

inFile = snakemake@input[["fname"]]
outFile = snakemake@output[["fname"]]

benchmark_df <- read.csv(inFile)
extras <- as.character(as.integer(sort(unique(benchmark_df$n_extras))))

# Color library
# colrys <- c("#ffc77c", "#CFB7A9", "#f2de8b", "#A7BBC9", "#F8766D", "#FF69B4", "#C77CFF", "#7cb4ff")
# colors <- c("#ffd8a3", "#ffc77c", "#ffb655",
#                "#ddccc2", "#CFB7A9", "#c1a290",
#                "#f6e8ae", "#f2de8b", "#eed468",
#                "#bfced8", "#A7BBC9", "#8fa8ba",
#                "#cae1d4", "#a9cfba", "#88bda0",
#                "#fa9992", "#F8766D", "#f65348",
#                "#ff90c8", "#FF69B4", "#ff42a0",
#                "#d8a3ff", "#C77CFF", "#b655ff",
#                "#a3caff", "#7cb4ff", "#559eff")

method_params <- list(correlation=list(name='Correlation', color=c("#ffd8a3", "#ffb655")),
                      clonealign=list(name='clonealign', color=c("#ddccc2", "#c1a290")),
                      leiden=list(name='leiden', color=c("#f6e8ae", "#eed468")),
                      clonealign_leiden=list(name='clonealign+leiden', color=c("#cae1d4", "#88bda0")),
                      scatrex=list(name='SCATrEx', color=c("#bfced8", "#8fa8ba")))
get_colors <- function(method_list, method_params_list) {
  colors <- list()
  i = 1
  for (method in names(method_params_list)){
    if (method %in% method_list) {
      colors <- append(colors, method_params_list[[method]]$color)
    }
    i <- i + 1
  }
  colors <- unlist(colors)
  colors
}

ptitle = 'Clone assignment accuracy'
score = 'clone_accuracy'
methods = c('correlation', 'clonealign', 'scatrex')
colors <- get_colors(methods, method_params)
benchmark_df_subset <- benchmark_df[benchmark_df$score == score,]
benchmark_df_subset <- benchmark_df_subset[benchmark_df_subset$method %in% methods,]
benchmark_df_subset$method <- factor(benchmark_df_subset$method , levels=methods)
p1 <- ggplot(benchmark_df_subset, aes(x = method, y = value, color = interaction(n_extras, method), fill = interaction(n_extras, method))) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitterdodge(), alpha=0.5) +
  xlab(NULL) +
  labs(y=ptitle) +
  scale_color_manual(values=colors, name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  scale_fill_manual(values=paste0(colors, "80"), name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  theme_bw(base_size = 16) + theme(legend.key.height = unit(1.5, "cm")) +
  theme(axis.text.x=element_text(angle=45,hjust=1,size=12)) +
  facet_wrap(~n_factors, labeller=label_bquote(paste(.(n_factors)," factors")),scales = "fixed", nrow = 1) #+

ptitle = 'Clone-level v-measure'
score = 'clone_ari'
methods = c('leiden', 'clonealign_leiden', 'scatrex')
colors <- get_colors(methods, method_params)
benchmark_df_subset <- benchmark_df[benchmark_df$score == score,]
benchmark_df_subset <- benchmark_df_subset[benchmark_df_subset$method %in% methods,]
benchmark_df_subset$method <- factor(benchmark_df_subset$method , levels=methods)
p2 <- ggplot(benchmark_df_subset, aes(x = method, y = value, color = interaction(n_extras, method), fill = interaction(n_extras, method))) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitterdodge(), alpha=0.5) +
  xlab(NULL) +
  labs(y=ptitle) +
  scale_color_manual(values=colors, name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  scale_fill_manual(values=paste0(colors, "80"), name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  theme_bw(base_size = 16) + theme(legend.key.height = unit(1.5, "cm")) +
  theme(axis.text.x=element_text(angle=45,hjust=1,size=12)) +
  facet_wrap(~n_factors, labeller=label_bquote(paste(.(n_factors)," factors")),scales = "fixed", nrow = 1) #+

ptitle = 'Node-level v-measure'
score = 'node_ari'
methods = c('leiden', 'clonealign_leiden', 'scatrex')
colors <- get_colors(methods, method_params)
benchmark_df_subset <- benchmark_df[benchmark_df$score == score,]
benchmark_df_subset <- benchmark_df_subset[benchmark_df_subset$method %in% methods,]
benchmark_df_subset$method <- factor(benchmark_df_subset$method , levels=methods)
p3 <- ggplot(benchmark_df_subset, aes(x = method, y = value, color = interaction(n_extras, method), fill = interaction(n_extras, method))) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitterdodge(), alpha=0.5) +
  xlab(NULL) +
  labs(y=ptitle) +
  scale_color_manual(values=colors, name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  scale_fill_manual(values=paste0(colors, "80"), name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  theme_bw(base_size = 16) + theme(legend.key.height = unit(1.5, "cm")) +
  theme(axis.text.x=element_text(angle=45,hjust=1,size=12)) +
  facet_wrap(~n_factors, labeller=label_bquote(paste(.(n_factors)," factors")),scales = "fixed", nrow = 1) #+

ptitle = 'Node assignment accuracy'
score = 'node_accuracy'
methods = c('scatrex')
colors <- get_colors(methods, method_params)
benchmark_df_subset <- benchmark_df[benchmark_df$score == score,]
benchmark_df_subset <- benchmark_df_subset[benchmark_df_subset$method %in% methods,]
benchmark_df_subset$method <- factor(benchmark_df_subset$method , levels=methods)
p4 <- ggplot(benchmark_df_subset, aes(x = method, y = value, color = interaction(n_extras, method), fill = interaction(n_extras, method))) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitterdodge(), alpha=0.5) +
  xlab(NULL) +
  labs(y=ptitle) +
  scale_color_manual(values=colors, name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  scale_fill_manual(values=paste0(colors, "80"), name= " \n ", labels=paste0(rep(methods, each=length(extras)), "\n", extras, " extra/clone")) +
  theme_bw(base_size = 16) + theme(legend.key.height = unit(1.5, "cm")) +
  theme(axis.text.x=element_text(angle=45,hjust=1,size=12)) +
  facet_wrap(~n_factors, labeller=label_bquote(paste(.(n_factors)," factors")),scales = "fixed", nrow = 1) #+

figure <- ggarrange(p1, p2, p3, p4,
                    labels = c("A", "B", "C", "D"),
                    ncol = 1, nrow = 4)
ggsave(outFile, width=15, height=16, dpi=200, bg = "transparent")
