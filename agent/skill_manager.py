import os
import re

class SkillManager:
    def __init__(self, skills_dir="agent/skills"):
        self.skills_dir = skills_dir
        self.skills = self._load_skills()

    def _load_skills(self):
        skills_data = {}
        if not os.path.exists(self.skills_dir):
            return {}
            
        for skill_id in os.listdir(self.skills_dir):
            md_path = os.path.join(self.skills_dir, skill_id, "SKILL.md")
            if os.path.exists(md_path):
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                desc_match = re.search(r'## Description\n(.*?)\n##', content, re.DOTALL)
                instr_match = re.search(r'## Instructions\n(.*)', content, re.DOTALL)
                
                skills_data[skill_id] = {
                    "id": skill_id,
                    "description": desc_match.group(1).strip() if desc_match else "",
                    "instructions": instr_match.group(1).strip() if instr_match else ""
                }
        return skills_data

    def get_skill_manifest(self):
        manifest = ""
        for s_id, data in self.skills.items():
            manifest += f"- SKILL_ID: {s_id}\n  DESCRIPTION: {data['description']}\n"
        return manifest