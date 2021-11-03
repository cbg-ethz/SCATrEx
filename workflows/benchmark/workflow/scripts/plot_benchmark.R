suppressWarnings(library(tidyverse))
suppressWarnings(library(ggpubr))

inFile = snakemake@input[["fname"]]
ariFile = snakemake@output[["ari_fname"]]
accFile = snakemake@output[["acc_fname"]]

benchmark_df <- read.csv(inFile)
extras <- as.character(as.integer(sort(unique(benchmark_df$n_extras)))+1)

method_params <- list(correlation=list(name='Correlation',newname='Correlation', color=c("#ffd8a3", "#ffb655", "#faa434")),
                      clonealign=list(name='Clonealign', newname='Clonealign', color=c("#ddccc2", "#c1a290", "#d69874")),
                      leiden=list(name='Leiden', newname='Leiden', color=c("#f6e8ae", "#eed468", "#f5cf38")),
                      clonealign_leiden=list(name='Clonealign+Leiden', newname='Clonealign+\nLeiden', color=c("#cae1d4", "#88bda0", "#5ed192")),
                      scatrex=list(name='SCATrEx', newname='SCATrEx', color=c("#bfced8", "#8fa8ba", "#699bbf")))
get_colors <- function(method_list, method_params_list) {
  colors <- list()
  i = 1
  for (method in method_list){
    if (method %in% names(method_params_list)) {
      colors <- append(colors, method_params_list[[method]]$color)
    }
    i <- i + 1
  }
  colors <- unlist(colors)
  colors
}

get_names <- function(method_list, method_params_list) {
  names <- list()
  i = 1
  for (method in method_list){
    if (method %in% names(method_params_list)) {
      names <- append(names, method_params_list[[method]]$name)
    }
    i <- i + 1
  }
  names <- unlist(names)
  names
}

get_newnames <- function(method_list, method_params_list) {
  names <- list()
  i = 1
  for (method in method_list){
    if (method %in% names(method_params_list)) {
      names <- append(names, method_params_list[[method]]$newname)
    }
    i <- i + 1
  }
  names <- unlist(names)
  names
}


all_methods <- list('correlation', 'clonealign', 'leiden', 'clonealign_leiden', 'scatrex')
all_colors <- get_colors(all_methods, method_params)
all_names <- get_names(all_methods, method_params)


ptitle = 'Clone assignment accuracy'
score = 'clone_accuracy'
methods = c('correlation', 'clonealign', 'scatrex')
names <- get_names(methods, method_params)
newnames <- get_newnames(methods, method_params)
colors <- get_colors(methods, method_params)
benchmark_df_subset <- benchmark_df[benchmark_df$score == score,]
benchmark_df_subset <- benchmark_df_subset[benchmark_df_subset$method %in% methods,]
benchmark_df_subset$method <- factor(benchmark_df_subset$method , levels=methods)
p1 <- ggplot(benchmark_df_subset, aes(x = method, y = value, color = interaction(n_extras, method), fill = interaction(n_extras, method))) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitterdodge(), alpha=0.6,size=2) +
  xlab(NULL) +
  labs(y=ptitle) +
  scale_x_discrete(labels=newnames) +
  scale_color_manual(values=colors, name= " \n ", labels=paste0(rep(names, each=length(extras)), "\n", extras, " nodes/clone"), drop=FALSE) +
  scale_fill_manual(values=paste0(colors, "80"), name= " \n ", labels=paste0(rep(names, each=length(extras)), "\n", extras, " nodes/clone"), drop=FALSE) +
  geom_hline(yintercept=.25, linetype="dashed", color = "gray") +
  guides(fill=guide_legend(ncol=2), color=guide_legend(ncol=2)) +
  theme_bw(base_size = 16) + theme(legend.key.height = unit(1.5, "cm")) +
  theme(axis.text.x=element_text(angle=45,hjust=1,size=12)) +
  facet_wrap(~n_factors, labeller=label_bquote(paste(.(n_factors)," noise factors")),scales = "fixed", nrow = 1) #+
ggsave(accFile, width=15, height=4.5, dpi=200, bg = "transparent", )

all_methods <- c('leiden', 'clonealign_leiden', 'scatrex')
all_colors <- get_colors(all_methods, method_params)
all_names <- get_names(all_methods, method_params)
ptitle = 'Node adjusted rand score'
score = 'node_ari'
methods = c('leiden', 'clonealign_leiden', 'scatrex')
names <- get_names(methods, method_params)
newnames <- get_newnames(methods, method_params)
colors <- get_colors(methods, method_params)
benchmark_df_subset <- benchmark_df[benchmark_df$score == score,]
benchmark_df_subset <- benchmark_df_subset[benchmark_df_subset$method %in% methods,]
benchmark_df_subset$method <- factor(benchmark_df_subset$method , levels=methods)
p2 <- ggplot(benchmark_df_subset, aes(x = method, y = value, color = interaction(n_extras, method), fill = interaction(n_extras, method))) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(shape=16, position=position_jitterdodge(), alpha=0.6, size=2) +
  xlab(NULL) +
  labs(y=ptitle) +
  scale_color_manual(values=colors, name= " \n ", labels=paste0(rep(newnames, each=length(extras)), "\n", extras, " nodes/clone")) +
  scale_fill_manual(values=paste0(colors, "80"), name= " \n ", labels=paste0(rep(newnames, each=length(extras)), "\n", extras, " nodes/clone")) +
  scale_x_discrete(labels=newnames) +
  theme_bw(base_size = 16) + theme(legend.key.height = unit(1.5, "cm")) +
  guides(fill=guide_legend(ncol=2), color=guide_legend(ncol=2)) +
  theme(axis.text.x=element_text(angle=45,hjust=1,size=12)) +
  facet_wrap(~n_factors, labeller=label_bquote(paste(.(n_factors)," noise factors")),scales = "fixed", nrow = 1) #+
ggsave(ariFile, width=15, height=4.5, dpi=200, bg = "transparent", )
