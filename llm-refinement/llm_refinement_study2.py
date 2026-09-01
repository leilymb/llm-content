# rate_topics_study2_urge.py
# Study 2 (urge) recategorization. Runs both a 14-topic version and a 15-topic
# version (14 + "Doesn't Fit" residual) side by side, mirroring the Study 1 setup.

import pandas as pd
import torch
import random
import numpy as np
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

#Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)

#Paths
MODEL_ID = "Qwen/Qwen3.6-27B"
CACHE_DIR = "/gpfs/radev/project/huggingface_qwen"
INPUT_PATH = "/gpfs/radev/project/working_env/urge_docs_2.csv"
OUTPUT_PATH = "/gpfs/radev/project/working_env/study2_urge_full_14vs15.csv"
CHECKPOINT_EVERY = 250

#Topic labels
TOPIC_LABELS_14 = {
    0: "Normal Life Thoughts",
    1: "Anxiety & Distress",
    2: "Suicide Method, Plan, or Preparations",
    3: "Low Self-Esteem & Worthlessness",
    4: "No Thoughts of Suicide",
    5: "Family & Friends Reactions",
    6: "Happy Life Thoughts",
    7: "Stressed & Overwhelmed",
    8: "Images of Self-harm, Suicide, & Death",
    9: "Passive Suicide Ideation",
    10: "Hopelessness",
    11: "Depressed & Exhausted",
    12: "Prefer not to respond",
    13: "Active Desire and/or Intent to Kill Oneself",
}
TOPIC_LABELS_15 = {**TOPIC_LABELS_14, 14: "Doesn't Fit Any Category"}

#Definitions
TOPIC_BLOCK_14 = """0. Normal Life Thoughts
Definition: Everyday, neutral, non-distressing content reflecting routine activities, daily concerns, or mundane observations. Responses in this category capture practical or ordinary content (e.g., school, work, errands, scheduling, casual social interactions) without strong emotional valence.
Example: "Everything like school and hobbies and cats"
Example: "I hope that I finish all my work today. I hope that I make a lot of money today. I need to take out the trash and clean up the house. I need to exercise and eat."

1. Anxiety & Distress
Definition: Emotional, somatic, or cognitive manifestations of anxiety or distress, including sensations such as feeling sick, tense, heavy, racing heart, anxious, jittery, or physically uncomfortable. It should be distinguished from stressed and overwhelmed, which involves cognitive content about appraising one's circumstances as overwhelming or unmanageable, rather than experiential or somatic sensations of anxiety itself, and from depressed and exhausted, which captures depressive affect and fatigue rather than anxiety-related sensations.
Example: "I'd feel out of breath and anxious. I would have to put my hands in my hair in attempts to stop my brain from thinking such negative thoughts. I'd also feel like I was incredibly alert with my surroundings, almost as if my eyes were wide the entirety of my spiraling."
Example: "I would probably be feeling sick, severe stomach ache, body shaking ands trembling, having trouble breathing."

2. Suicide Method, Plan, or Preparations
Definition: Thoughts focused on the methods, plans, preparations, or logistics of suicide. Content centers on how one would carry out an act of suicide, including specific methods, thinking about methods, planning steps, suicide notes, or preparations like saying goodbye to loved ones. It should be distinguished from images of self-harm and suicide methods where the content is visual imagery rather than active planning, and from passive suicidal ideation, which lacks any action orientation.
Example: "Trying to find a gun, pills or a blade I could use to kill myself"
Example: "I would be having thoughts about what and how I would kill myself. I would wonder if one thing would hurt more than the other and how quickly"

3. Low Self-Esteem & Worthlessness
Definition: Negative self-evaluative cognitions, including self-criticism, self-hatred, perceived worthlessness, or feeling like a burden. Thoughts about feeling worthless and less than others, or a waste of space. It should be distinguished from depressed and exhausted, which is general depressive affect without specifically self-evaluative content, and from family and friends reactions, which focus on others' grief or response, rather than self-criticism.
Example: "You will always be failure"
Example: "Who would care"

4. No Thoughts of Suicide
Definition: Responses that explicitly deny the presence of suicidal thoughts or imagery.
Example: "none"
Example: "None? I don't think about suicide until I'm experiencing some negative emotion."
Example: "No images"

5. Family & Friends Reactions
Definition: Thoughts about how one's death by suicide would affect significant others, including family, friends, partners, or pets. Responses might include anticipated grief or pain experienced by loved ones, funeral imagery, fears of traumatizing others (e.g., being found by family), or worries about leaving pets behind.
Example: "my animals would miss me"
Example: "my friends and families reaction to me dying"

6. Happy Life Thoughts
Definition: Positive, happy, joyful, hopeful, or protective content about one's life, including reasons to live, future-oriented optimism, and enjoyment of life. It should be distinguished from normal life thoughts, which are neutral, mundane content without strong affective valence, and from family and friends reactions when those are framed in the context of suicide impact rather than positive connection.
Example: "Fun loving moments"
Example: "images of living happily in the future, with friends and maybe a partner, helping people"

7. Stressed & Overwhelmed
Definition: Thoughts and emotions reflecting feelings of being overwhelmed, stressed, or unable to manage one's circumstances, including a sense that things are going wrong or beyond one's control.
Example: "It usually happens when I have a lot of stress happening in my life or if I feel out of control of the things happening to me/my situation. A lot of the time I feel backed against a wall and feel trapped."
Example: "Probably a lot of feelings of stress and overwhelm and just not wanting to talk to anyone or do anything."

8. Images of Self-harm, Suicide, & Death
Definition: Visual mental imagery depicting self-harm, methods of suicide, or one's own death. This could include blood, wounds, scenes of suicidal acts, seeing oneself die/dead. The content is the visual experience of imagery rather than active cognitive planning. It should be distinguished from thinking of a method and or a plan, which involves verbal or cognitive method consideration, and from family and friends reactions, where imagery is focused on others rather than on the act itself.
Example: "the image of me dying and the things that could kill me"
Example: "I would see images of myself executing my plan to kill myself, taking my last breath, hoping I didn't make the wrong decision, and going into the afterlife."

9. Passive Suicide Ideation
Definition: Thoughts reflecting a wish to be dead, to not exist, or for life to end, in the absence of active intent, plan, or motivation to take one's life. Passive ideation is distinguished from active ideation by the absence of action orientation; the individual desires death but is not contemplating doing something to cause it. It should be distinguished from active suicidal ideation and thinking of a method and or a plan, which both involve intent or planning, and from ruminating about death, which involves curiosity about death without a personal wish to die.
Example: "I'd want to die, but not necessarily have a plan to nor the intent to act on it myself"
Example: "I would passively be thinking about dying and not being here anymore. I would probably think that I don't want to exist."

10. Hopelessness
Definition: Thoughts reflecting a belief that one's situation, future, or life will not improve, including a sense that life is pointless or that there is no path forward. It should be distinguished from depressed and exhausted, which involves depressive affect and somatic symptoms rather than future-oriented content, from low self-esteem and worthlessness, which involves negative self-evaluation rather than negative future orientation, and from passive suicidal ideation, which involves a wish for death rather than a belief that life cannot improve.
Example: "I would just feel pretty hopeless and that there's no way out of my situation. I would not feel like actively ending it but I would be building up to that point."
Example: "I would likely be asking myself why I am still putting up with the problems in my life. Why am I still trying when things look so bleak? What's the point in going on living?"

11. Depressed & Exhausted
Definition: Responses sharing affect and somatic content reflecting depressive states, including sadness, fatigue, anhedonia, crying, feelings of numbness, sleep disturbance or related distress. These thoughts should be distinguished from low self-esteem, which specifically involves self-evaluative negative thoughts, and from any suicide-related category.
Example: "How life sucks and how I don't feel motivated to do anything and how tired I am."
Example: "I would probably feel depressed, hopeless, embarrassed, and mad at myself for some reason."

12. Prefer not to respond
Definition: Responses indicating that the participant declines to answer or chooses not to provide content for the given rating. The defining feature is explicit opt-out rather than substantive content or interpretive uncertainty. It should be distinguished from no thoughts about suicide, which is an explicit denial of suicide-specific content.
Example: "prefer not to answer"
Example: "prefer not to respond"

13. Active Desire and/or Intent to Kill Oneself
Definition: Thoughts reflecting active desire, intent, or urgency to die by suicide, going beyond a wish to be dead to include orientation toward acting on that wish. Active ideation is distinguished from passive ideation by the presence of intent or motivation toward action, even when a specific method or plan is not yet articulated. It should be distinguished from thinking of a method and/or plan, which adds method or plan-specific content, and from passive suicidal ideation, which lacks action orientation.
Example: "I should kill myself next week"
Example: "At a 10, I would be actively trying to kill myself. I am actively doing something with the intention of ending my life."
"""

TOPIC_15_BLOCK = """
14. Doesn't Fit Any Category
Definition: Responses whose content does not meaningfully fit into any of the above 14 topics. This includes off-topic content, content unrelated to the prompt about suicidal urge (e.g., gibberish, copy-pasted survey instructions, responses that appear to be misinterpretations of the question), or content too incoherent to map to any defined topic. Use this ONLY when no other category genuinely applies — not for ambiguous cases that could plausibly fit one of the existing 14 topics. It should be distinguished from No Thoughts of Suicide, which is an explicit denial of suicidal thoughts, and from Prefer not to respond, which is an explicit opt-out.
Example: "Write a minimum of 90 characters."
Example: "asdkfjasldkfj"
"""

SYSTEM_MSG_14 = """You are an expert in psychology and suicidology. You will categorize a participant's response into ONE of 14 topics related to thoughts about suicidal urge.

Each response describes what a participant imagined they would be thinking — including suicide-related thoughts, unrelated everyday thoughts, and mental imagery — when imagining themselves at a hypothetical level of suicidal urge.

Each topic includes a definition (with inclusion/exclusion criteria) and two or three example responses. Choose the SINGLE most appropriate topic based on the psychological/emotional content described.

Important guidelines:
- Focus on what the response describes, not surface words or keywords alone.
- If a response mentions numerical ratings (e.g., "at a 5", "rating of 8"), those numbers are NOT topic numbers. Ignore them.
- Use the exclusion criteria to resolve cases that look similar to multiple topics.

Respond with ONLY the topic number (0-13). No other text, no explanation."""

SYSTEM_MSG_15 = """You are an expert in psychology and suicidology. You will categorize a participant's response into ONE of 15 topics related to thoughts about suicidal urge.

Each response describes what a participant imagined they would be thinking — including suicide-related thoughts, unrelated everyday thoughts, and mental imagery — when imagining themselves at a hypothetical level of suicidal urge.

Each topic includes a definition (with inclusion/exclusion criteria) and two or three example responses. Choose the SINGLE most appropriate topic based on the psychological/emotional content described.

Important guidelines:
- Focus on what the response describes, not surface words or keywords alone.
- If a response mentions numerical ratings (e.g., "at a 5", "rating of 8"), those numbers are NOT topic numbers. Ignore them.
- Use the exclusion criteria to resolve cases that look similar to multiple topics.
- Topic 14 ("Doesn't Fit Any Category") is a LAST RESORT — only use it when no other topic genuinely applies. Do not use it for ambiguous cases that could plausibly fit one of topics 0-13.

Respond with ONLY the topic number (0-14). No other text, no explanation."""

#Model loading
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

#Prompt construction
def build_prompt(response_text, version):
    if version == 14:
        system_msg = SYSTEM_MSG_14
        topic_block = TOPIC_BLOCK_14
    else:  # 15
        system_msg = SYSTEM_MSG_15
        topic_block = TOPIC_BLOCK_14 + TOPIC_15_BLOCK

    user_msg = (
        f"Topics:\n{topic_block}\n"
        f"Response to categorize:\n\"{response_text.strip()}\"\n\n"
        f"Topic number:"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

#Inference 
def get_topic(response_text, fallback_topic, version):
    prompt = build_prompt(response_text, version)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    valid = set(range(version))  # 0-13 or 0-14
    for tok in re.findall(r"\d+", response):
        if int(tok) in valid:
            return int(tok), False, response

    return int(fallback_topic), True, response

#Main
def main():
    df = pd.read_csv(INPUT_PATH)[["Document", "Topic"]].copy()
    df["Document"] = df["Document"].fillna("").astype(str)

    results = []
    start = time.time()

    for idx, row in df.iterrows():
        topic_14, fail_14, raw_14 = get_topic(row["Document"], row["Topic"], version=14)
        topic_15, fail_15, raw_15 = get_topic(row["Document"], row["Topic"], version=15)

        results.append({
            "Document": row["Document"],
            "bertopic_Topic": row["Topic"],
            "topic_14": topic_14,
            "topic_14_label": TOPIC_LABELS_14.get(topic_14),
            "topic_14_parse_failed": fail_14,
            "topic_14_raw": raw_14,
            "topic_15": topic_15,
            "topic_15_label": TOPIC_LABELS_15.get(topic_15),
            "topic_15_parse_failed": fail_15,
            "topic_15_raw": raw_15,
            "agree": topic_14 == topic_15,
            "landed_in_residual": topic_15 == 14,
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta = (len(df) - (idx + 1)) / rate if rate > 0 else 0
            print(
                f"Processed {idx+1}/{len(df)} | "
                f"elapsed: {elapsed:.0f}s | "
                f"rate: {rate:.2f}/s | "
                f"ETA: {eta:.0f}s ({eta/60:.1f} min)"
            )

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            ckpt = pd.DataFrame(results)
            ckpt_path = OUTPUT_PATH.replace(".csv", f"_ckpt_{idx+1}.csv")
            ckpt.to_csv(ckpt_path, index=False)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)

    # Summary
    n = len(out)
    n_disagree = (~out["agree"]).sum()
    n_residual = out["landed_in_residual"].sum()
    n_fail_14 = out["topic_14_parse_failed"].sum()
    n_fail_15 = out["topic_15_parse_failed"].sum()

    print("\n" + "=" * 60)
    print(f"STUDY 2 URGE - FULL SUMMARY (n = {n})")
    print("=" * 60)
    print(f"Disagreement between 14- and 15-topic versions: {n_disagree}/{n} ({100*n_disagree/n:.1f}%)")
    print(f"Landed in 'Doesn't Fit Any Category' (15-version): {n_residual}/{n} ({100*n_residual/n:.1f}%)")
    print(f"Parse failures (14-topic): {n_fail_14}/{n} ({100*n_fail_14/n:.1f}%)")
    print(f"Parse failures (15-topic): {n_fail_15}/{n} ({100*n_fail_15/n:.1f}%)")
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Total time: {(time.time() - start):.0f}s ({(time.time() - start)/60:.1f} min)")

    print("\n--- 14-topic distribution ---")
    print(out["topic_14_label"].value_counts().to_string())
    print("\n--- 15-topic distribution ---")
    print(out["topic_15_label"].value_counts().to_string())

if __name__ == "__main__":
    main()
