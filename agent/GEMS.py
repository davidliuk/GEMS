import base64
import os
import re
import json
from openai import OpenAI
from agent.base_agent import BaseAgent
from concurrent.futures import ThreadPoolExecutor

SUMMARIZE_EXPERIENCE_TEMPLATE = (
    "Task: Summarize the experience of the current image generation attempt.\n"
    "--- CURRENT ATTEMPT ---\n"
    "Prompt used: {current_prompt}\n"
    "Passed requirements: {passed}\n"
    "Failed requirements: {failed}\n"
    "Reasoning/Thought before generation: {current_thought}\n"
    "Image: <image>\n" 
    "--- PREVIOUS EXPERIENCES ---\n"
    "{previous_experiences}\n"
    "--- ANALYSIS ---\n"
    "Based on the provided image, the verification results, your previous thought process, and historical experiences, "
    "write a concise summary (experience) of what worked, what failed, and what strategy should be adopted in the next attempt. "
    "Keep it under 100 words. Do not include introductory phrases."
)

DECOMPOSE_PROMPT = (
    "Analyze the user's image generation prompt. "
    "Break it down into specific visual requirements. "
    "For each requirement, write a question that can be answered with a simple 'yes' or 'no'. "
    "The questions should verify if the requirement is met in an image. "
    "YOU MUST RESPOND ONLY WITH A JSON ARRAY OF STRINGS. "
    "Example format: [\"Is there a cat?\", \"Is the cat black?\", \"Is it sitting on a rug?\"]"
)

VERIFY_PROMPT_PREFIX = (
    "Answer the following question with only 'yes' or 'no' based on the provided image: "
)

REFINE_PROMPT_TEMPLATE = (
    "Task: Refine the image generation prompt based on previous failed attempts and accumulated experiences.\n"
    "Original Intent: {original_prompt}\n\n"
    "--- ATTEMPT HISTORY ---\n"
    "{history_log}\n"
    "--- ANALYSIS ---\n"
    "Review the history above. Rewrite a new, comprehensive prompt. This prompt must:\n"
    "1. Explicitly reinforce the requirements that failed in the latest attempt.\n"
    "2. Maintain and protect the requirements that were successfully met in previous rounds to avoid regressions.\n"
    "3. Adopt the strategies suggested in the 'Experience' section.\n"
    "4. Use clear, non-conflicting descriptive language.\n\n"
    "Return ONLY the prompt text itself. Do not include any conversational filler, introductory remarks, "
    "or prefixes like 'New prompt:' or 'Refined prompt:'."
)

PLANNER_DECISION_PROMPT = (
    "You are a strategic Skill Router. Your goal is to determine if the user's request "
    "genuinely requires a specialized skill or if it can be handled by standard generation.\n\n"
    "### Available Skills:\n{manifest}\n"
    "### User Request:\n{user_prompt}\n\n"
    "### Evaluation Criteria:\n"
    "1. **Relevance**: Does the request explicitly match the 'DESCRIPTION' of a skill?\n"
    "2. **Added Value**: Does using this skill provide significant benefits (e.g., specific artistic styles, "
    "complex logic, or reference handling) that standard generation lacks?\n"
    "3. **Default to NONE**: If the request is simple, generic, or does not strongly align with any skill, "
    "you MUST choose 'NONE'.\n\n"
    "### Response Requirement:\n"
    "- If a skill is a strong match, respond ONLY with the SKILL_ID.\n"
    "- Otherwise, respond ONLY with 'NONE'.\n"
    "Do not provide any explanation, preamble, or punctuation."
)

class GEMS(BaseAgent):
    def __init__(self, gen_url, mllm_url, max_iterations):
        super().__init__(gen_url, mllm_url)
        self.max_iterations = max_iterations

    def decompose(self, prompt: str) -> list:
        task = f"{DECOMPOSE_PROMPT}\n\nUser Prompt: {prompt}"
        response = self.think(task).strip()
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group()
                questions = json.loads(json_str)
            else:
                questions = json.loads(response)
                
            if isinstance(questions, list):
                return [q for q in questions if isinstance(q, str)]
            return []
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"[Error] JSON parsing failed, returning raw line-split results. Error: {e}")
            return [line.strip() for line in response.split('\n') if '?' in line]

    def verify_image(self, image_bytes: bytes, questions: list) -> list:
        def ask_question(q):
            full_query = f"Image: <image>\n{VERIFY_PROMPT_PREFIX} {q}"
            try:
                answer = self.think(full_query, images=[image_bytes]).lower().strip()
                is_passed = "yes" in answer and "no" not in answer
                return {"question": q, "answer": answer, "passed": is_passed}
            except Exception as e:
                return {"question": q, "answer": f"Error: {e}", "passed": False}

        with ThreadPoolExecutor(max_workers=min(len(questions), 10)) as executor:
            results = list(executor.map(ask_question, questions))
        return results

    def plan(self, original_prompt: str) -> str:
        manifest = self.skill_manager.get_skill_manifest()
        
        decision_task = PLANNER_DECISION_PROMPT.format(
            manifest=manifest,
            user_prompt=original_prompt
        )
        skill_id = self.think(decision_task).strip()
        
        if skill_id in self.skill_manager.skills:
            print(f"🎯 LLM decided to trigger skill: {skill_id}")
            skill_info = self.skill_manager.skills[skill_id]
            
            refine_task = (
                f"Based on the following skill instructions, enhance the user's prompt.\n"
                f"### Skill Instructions:\n{skill_info['instructions']}\n\n"
                f"### Original Prompt: {original_prompt}\n\n"
                f"Return ONLY the final enhanced prompt text."
            )
            enhanced_prompt = self.think(refine_task).strip()
            print(f"🚀 Enhanced Prompt: {enhanced_prompt}")
            return enhanced_prompt
        
        print("⏭️ No skill triggered, using original prompt.")
        return original_prompt

    def run(self, item: dict) -> bytes:
        original_prompt = item.get("prompt", "")
        
        print(f"\n[Step 0] Skill Routing and Planning...")
        current_prompt = self.plan(original_prompt)
        
        if current_prompt != original_prompt:
            current_thought = "Initial submission enhanced by specialized skill instructions."
        else:
            current_thought = "Initial submission based on original prompt."
        
        print(f"\n[Step 1] Decomposing requirements...")

        questions = self.decompose(original_prompt)
        
        if not questions:
            return self.generate(original_prompt)

        attempt_history = []
        best_image_bytes = None
        max_passed_count = -1
        
        for i in range(1, self.max_iterations + 1):
            print(f"\n--- Round {i}/{self.max_iterations} ---")
            
            final_image_bytes = self.generate(current_prompt)
            
            verifications = self.verify_image(final_image_bytes, questions)
            failed_questions = [v['question'] for v in verifications if not v['passed']]
            passed_questions = [v['question'] for v in verifications if v['passed']]

            current_passed_count = len(passed_questions)
            if current_passed_count > max_passed_count:
                max_passed_count = current_passed_count
                best_image_bytes = final_image_bytes
                print(f" [Updating best solution] Current passed: {max_passed_count}/{len(questions)}")
                
            for v in verifications:
                print(f"  {'✅' if v['passed'] else '❌'} {v['question']}")

            if not failed_questions:
                print(f"\nSuccess: All requirements met!")
                return final_image_bytes

            if i < self.max_iterations:
                print(f" [Extracting experience] Compressing Thought into Experience...")
                previous_experiences = "\n".join([f"Round {record['iteration']}: {record['experience']}" for record in attempt_history])
                if not previous_experiences:
                    previous_experiences = "None (First round)"

                summarize_task = SUMMARIZE_EXPERIENCE_TEMPLATE.format(
                    current_prompt=current_prompt,
                    passed=", ".join(passed_questions) if passed_questions else "None",
                    failed=", ".join(failed_questions) if failed_questions else "None",
                    current_thought=current_thought,
                    previous_experiences=previous_experiences
                )

                current_experience = self.think(summarize_task, images=[final_image_bytes]).strip()
                print(f" Current round experience: {current_experience}")

                attempt_history.append({
                    "iteration": i,
                    "prompt": current_prompt,
                    "experience": current_experience, 
                    "failed": failed_questions,
                    "passed": passed_questions,
                    "image_bytes": final_image_bytes
                })

                print(f"Optimizing prompt based on historical images...")
                
                history_log_str = ""
                history_images = []
                
                for record in attempt_history:
                    history_log_str += (
                        f"Attempt {record['iteration']}:\n"
                        f"- Experience: {record['experience']}\n" 
                        f"- Prompt: {record['prompt']}\n"
                        f"- Image Result: <image>\n"
                        f"- Failed Points: {', '.join(record['failed']) if record['failed'] else 'None'}\n\n"
                    )
                    history_images.append(record['image_bytes'])

                refine_task = REFINE_PROMPT_TEMPLATE.format(
                    original_prompt=original_prompt,
                    history_log=history_log_str
                )
                
                current_prompt, current_thought = self.think_with_thought(refine_task, images=history_images)
                current_prompt = current_prompt.strip()
                print(f"Newly generated Prompt: {current_prompt}")
                print(f"Newly generated Thought: {current_thought}")
            else:
                print(f"Maximum iterations reached.")

        print(f"Returning the best image from iterations (Passed: {max_passed_count}/{len(questions)}).")
        return best_image_bytes