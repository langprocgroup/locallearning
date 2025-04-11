rm(list=ls())
setwd("~/projects/locallearning")
library(tidyverse)
library(plotrix)
library(latex2exp)


noiseless = read_csv("data/k3_p51_noiseless.csv") %>%
  mutate(lam="0", p=".51") %>%
  bind_rows(read_csv("data/k3_p65_noiseless.csv") %>%
  mutate(lam="0", p=".79") %>%
  bind_rows(read_csv("data/k3_p79_noiseless.csv") %>%
  mutate(lam="0", p=".79")))

noisy = read_csv("data/k3_p75_n99.csv") %>%
  mutate(lam=".99", p=".75") %>%
  bind_rows(read_csv("data/k3_p75_n9.csv") %>%
  mutate(lam=".9", p=".75") %>%
  bind_rows(read_csv("data/k3_p75_n5.csv") %>%
  mutate(lam=".5", p=".75")))

redundant = read_csv("data/k3_n9_p79_red2.csv")

noiseless %>% 
  group_by(target, i, p) %>% 
    summarize(se=std.error(entropy), 
              entropy=mean(entropy),
              upper=entropy+1.96*se,
              lower=entropy-1.96*se) %>%
    ungroup() %>% 
  bind_rows(data.frame(i=-1, p=".51", entropy=log(factorial(2^3)))) %>%
  bind_rows(data.frame(i=-1, p=".65", entropy=log(factorial(2^3)))) %>%
  bind_rows(data.frame(i=-1, p=".79", entropy=log(factorial(2^3)))) %>%
  #mutate(entropy=if_else(i==0, log(factorial(8)), entropy)) %>%
  mutate(p=case_when(
    p == ".51" ~ "b = .51, H[M] = 2.8 bits",
    p == ".65" ~ "b = .65, H[M] = 2.4 bits",
    p == ".79" ~ "b = .79, H[M] = 1.3 bits",
    FALSE ~ "hi!"
  )) %>%
  ggplot(aes(x=i+2, y=entropy/log(2), color=p)) + 
    geom_line() +
    theme_classic() +
    scale_x_log10() +
    annotation_logticks(sides="b") +
    labs(x="Form Observations", y="Key entropy H[K] (bits)") +
    theme(legend.title=element_blank(),
          legend.position=c(.75,.74))

ggsave("learning_noiseless.pdf", width=4, height=3)


noisy %>% 
  group_by(target, i, lam) %>% 
    summarize(se=std.error(entropy), 
              entropy=mean(entropy),
              upper=entropy+1.96*se,
              lower=entropy-1.96*se) %>%
    ungroup() %>% 
  mutate(type=if_else(target == 0 | target == 37965, "compositional", "holistic")) %>%
  mutate(noise=case_when(
    lam == ".5" ~ "Noise rate e = .5",
    lam == ".9" ~ "Noise rate e = .9",
    lam == ".99" ~ "Noise rate e = .99"
  )) %>%
  ggplot(aes(x=i+1, y=entropy/log(2), linetype=type, color=as.factor(target))) + 
  geom_line() +
  facet_wrap(~noise) +
  theme_classic() +
  scale_x_log10() +
  annotation_logticks(sides="b") +
  theme(legend.position=c(.47,.3)) +
  labs(color="", linetype="") +
  guides(color="none") +
  ylim(0, NA) +
  labs(x="Form Observations", y="Key entropy H[K] (bits)") 

ggsave("learning_noisy.pdf", width=8, height=2.5)

redundant %>%
  group_by(target, i) %>%
    summarize(se=std.error(entropy), 
              entropy=mean(entropy),
              upper=entropy+1.96*se,
              lower=entropy-1.96*se) %>%
    ungroup() %>% 
  mutate(type=if_else(target == 0, "local", "nonlocal")) %>%
  ggplot(aes(x=i+1, y=entropy/log(2), color=type)) +
    geom_line() +
    theme_classic() +
    scale_x_log10() +
    annotation_logticks(sides="b") +
    theme(legend.position=c(.35,.35)) +
    labs(color="", x="Form Observations", y="Key entropy H[K] (bits)") +
    ylim(0, NA)
  
ggsave("learning_redundant.pdf", width=4, height=3)

