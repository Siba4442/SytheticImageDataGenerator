#!/usr/bin/env python3
"""
Enhanced script to generate realistic multiline Odia text images
with paragraph, poem, and document-like structures
"""

from PIL import Image, ImageDraw, ImageFont
import os
import textwrap
import random

class OdiaTextImageGenerator:
    def __init__(self):
        self.font_paths = [
            r"C:\Users\sibap\Programming\Odia_gen_ai\VLM-synthetic-data-generation\fonts\NotoSansOriya.ttf",
            "C:/Windows/Fonts/NotoSansOriya-Regular.ttf",
            "C:/Windows/Fonts/kalinga.ttf",
            "/System/Library/Fonts/NotoSansOriya.ttf",  # macOS
            "/usr/share/fonts/truetype/noto/NotoSansOriya-Regular.ttf"  # Linux
        ]
        
    def load_font(self, size=30):
        """Load Odia font with fallback options"""
        for path in self.font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    print(f"Using font: {path}")
                    return font
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        
        print("Using default font - Odia text may not render properly")
        return ImageFont.load_default()
    
    def create_paragraph_document(self):
        """Create a document-style image with multiple paragraphs"""
        
        # Sample Odia paragraphs
        paragraphs = [
            "ଓଡ଼ିଶା ପୂର୍ବ ଭାରତର ଏକ ସୁନ୍ଦର ରାଜ୍ୟ। ଏଠାରେ ଅନେକ ପ୍ରାଚୀନ ମନ୍ଦିର ଓ ସାଂସ୍କୃତିକ ସ୍ଥାନ ରହିଛି। ଜଗନ୍ନାଥ ମନ୍ଦିର ପୁରୀରେ ଅବସ୍ଥିତ ଏବଂ ଏହା ବିଶ୍ୱ ପ୍ରସିଦ୍ଧ।",
            
            "ଓଡ଼ିଆ ଭାଷା ଏକ ଇଣ୍ଡୋ-ଆର୍ଯ୍ୟ ଭାଷା ଅଟେ। ଏହାର ନିଜସ୍ୱ ଲିପି ଓ ସମୃଦ୍ଧ ସାହିତ୍ୟ ପରମ୍ପରା ରହିଛି। ପ୍ରାଚୀନ କାଳରୁ ଆଜି ପର୍ଯ୍ୟନ୍ତ ଅନେକ ମହାନ କବି ଓ ଲେଖକ ଏହି ଭାଷାରେ ଲେଖିଛନ୍ତି।",
            
            "କୋଣାର୍କର ସୂର୍ଯ୍ୟ ମନ୍ଦିର ଓଡ଼ିଶାର ଗର୍ବ। ଏହା ତେରଶତମ ଶତାବ୍ଦୀରେ ନିର୍ମିତ ହୋଇଥିଲା। ଏହାର ସ୍ଥାପତ୍ୟ ଓ ଶିଳ୍ପକଳା ସମଗ୍ର ବିଶ୍ୱରେ ପ୍ରଶଂସିତ।"
        ]
        
        # Create image
        img_width = 800
        img_height = 900
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        title_font = self.load_font(45)
        body_font = self.load_font(32)
        
        # Add title
        title = "ଓଡ଼ିଶାର ଐତିହ୍ୟ"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (img_width - title_width) // 2
        draw.text((title_x, 40), title, font=title_font, fill='darkblue')
        
        # Add underline
        draw.line([(title_x, 100), (title_x + title_width, 100)], fill='darkblue', width=2)
        
        # Add paragraphs
        y_position = 150
        margin = 50
        
        for i, paragraph in enumerate(paragraphs):
            # Wrap text to fit within margins
            lines = self.wrap_odia_text(paragraph, body_font, img_width - 2*margin)
            
            for line in lines:
                draw.text((margin, y_position), line, font=body_font, fill='black')
                y_position += 45
            
            # Add space between paragraphs
            y_position += 25
        
        # Save
        output_file = "odia_document.png"
        img.save(output_file)
        print(f"Created document-style image: {output_file}")
        return output_file
    
    def create_poem_image(self):
        """Create a poem-style image with centered verses"""
        
        # Sample Odia poem
        poem_lines = [
            "ମୋର ଭାରତ ଭୂମି",
            "",
            "ମୋର ଭାରତ ଭୂମି ମା' ତୁମେ ମହାନ",
            "ତୁମର ମାଟିରେ ଜନ୍ମି ମୁଁ ଗର୍ବିତ ପ୍ରାଣ",
            "ଗଙ୍ଗା ଯମୁନା ତୁମର ପବିତ୍ର ନଦୀ",
            "ହିମାଳୟ ତୁମର ମୁକୁଟ ଶୋଭା ବଢ଼ି",
            "",
            "ଅନେକ ଭାଷା ଅନେକ ଧର୍ମ",
            "ସବୁ ମିଶି ଏକ ଭାରତୀୟ ମର୍ମ",
            "ଏକତାରେ ବିବିଧତା ତୁମର ଶକ୍ତି",
            "ଜୟ ହିନ୍ଦ ଜୟ ଭାରତ ମାତା ଭକ୍ତି"
        ]
        
        # Create image
        img_width = 700
        img_height = 600
        img = Image.new('RGB', (img_width, img_height), '#f8f8f0')
        draw = ImageDraw.Draw(img)
        
        # Add decorative border
        border_color = 'darkgreen'
        draw.rectangle([(10, 10), (img_width-10, img_height-10)], outline=border_color, width=3)
        draw.rectangle([(20, 20), (img_width-20, img_height-20)], outline=border_color, width=1)
        
        # Load fonts
        title_font = self.load_font(40)
        poem_font = self.load_font(28)
        
        # Start positioning
        y_position = 50
        
        for i, line in enumerate(poem_lines):
            if i == 0:  # Title
                # Center the title
                bbox = draw.textbbox((0, 0), line, font=title_font)
                line_width = bbox[2] - bbox[0]
                x_position = (img_width - line_width) // 2
                draw.text((x_position, y_position), line, font=title_font, fill='darkred')
                y_position += 60
            elif line == "":  # Empty line for spacing
                y_position += 20
            else:  # Poem lines
                # Center each line
                bbox = draw.textbbox((0, 0), line, font=poem_font)
                line_width = bbox[2] - bbox[0]
                x_position = (img_width - line_width) // 2
                draw.text((x_position, y_position), line, font=poem_font, fill='darkblue')
                y_position += 40
        
        # Save
        output_file = "odia_poem.png"
        img.save(output_file)
        print(f"Created poem-style image: {output_file}")
        return output_file
    
    def create_news_article(self):
        """Create a newspaper-style article"""
        
        headline = "ପୁରୀ ଜଗନ୍ନାଥ ମନ୍ଦିରରେ ଆଜି ବିଶେଷ ପୂଜା"
        
        article_text = """ପୁରୀ, ଜୁନ ୧୫: ଆଜି ପୁରୀ ଜଗନ୍ନାଥ ମନ୍ଦିରରେ ବିଶେଷ ପୂଜା ଅନୁଷ୍ଠିତ ହେବ। ସକାଳ ୬ଟାରୁ ସନ୍ଧ୍ୟା ୮ଟା ପର୍ଯ୍ୟନ୍ତ ଏହି ପୂଜା ଚାଲିବ।
        
ମନ୍ଦିର ପ୍ରଶାସନ ସୂଚନା ଦେଇଛନ୍ତି ଯେ ଆଜି ହଜାର ହଜାର ଭକ୍ତ ଦର୍ଶନ ପାଇଁ ଆସିବେ। ବିଶେଷ ସୁରକ୍ଷା ବ୍ୟବସ୍ଥା କରାଯାଇଛି।
        
ଜଗନ୍ନାଥ, ବଳଭଦ୍ର ଓ ସୁଭଦ୍ରାଙ୍କ ବିଶେଷ ସାଜସଜ୍ଜା କରାଯାଇଛି। ମହାପ୍ରସାଦ ବଣ୍ଟନ ମଧ୍ୟ ହେବ।"""
        
        # Create image
        img_width = 750
        img_height = 800
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        headline_font = self.load_font(38)
        date_font = self.load_font(24)
        body_font = self.load_font(28)
        
        # Add newspaper header
        header = "ଦୈନିକ ସମାଚାର"
        header_bbox = draw.textbbox((0, 0), header, font=date_font)
        header_width = header_bbox[2] - header_bbox[0]
        draw.text(((img_width - header_width) // 2, 20), header, font=date_font, fill='black')
        
        # Add date
        date_text = "ରବିବାର, ଜୁନ ୧୫, ୨୦୨୫"
        date_bbox = draw.textbbox((0, 0), date_text, font=date_font)
        date_width = date_bbox[2] - date_bbox[0]
        draw.text(((img_width - date_width) // 2, 50), date_text, font=date_font, fill='gray')
        
        # Add separator line
        draw.line([(50, 85), (img_width-50, 85)], fill='black', width=2)
        
        # Add headline
        headline_lines = self.wrap_odia_text(headline, headline_font, img_width - 100)
        y_pos = 110
        for line in headline_lines:
            line_bbox = draw.textbbox((0, 0), line, font=headline_font)
            line_width = line_bbox[2] - line_bbox[0]
            draw.text(((img_width - line_width) // 2, y_pos), line, font=headline_font, fill='darkred')
            y_pos += 50
        
        # Add separator
        y_pos += 20
        draw.line([(100, y_pos), (img_width-100, y_pos)], fill='darkred', width=1)
        y_pos += 30
        
        # Add article body
        paragraphs = article_text.strip().split('\n        \n')
        margin = 60
        
        for paragraph in paragraphs:
            if paragraph.strip():
                lines = self.wrap_odia_text(paragraph.strip(), body_font, img_width - 2*margin)
                for line in lines:
                    draw.text((margin, y_pos), line, font=body_font, fill='black')
                    y_pos += 38
                y_pos += 15  # Paragraph spacing
        
        # Save
        output_file = "odia_news.png"
        img.save(output_file)
        print(f"Created news article image: {output_file}")
        return output_file
    
    def create_letter_format(self):
        """Create a formal letter in Odia"""
        
        # Create image
        img_width = 700
        img_height = 900
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Load font
        font = self.load_font(26)
        
        # Letter content
        y_pos = 50
        margin = 60
        
        # Date
        draw.text((img_width - 200, y_pos), "ତାରିଖ: ୧୫/୬/୨୦୨୫", font=font, fill='black')
        y_pos += 80
        
        # Address
        draw.text((margin, y_pos), "ପ୍ରତି,", font=font, fill='black')
        y_pos += 40
        draw.text((margin + 20, y_pos), "ମୁଖ୍ୟ ଶିକ୍ଷକ ମହୋଦୟ", font=font, fill='black')
        y_pos += 35
        draw.text((margin + 20, y_pos), "ସରକାରୀ ଉଚ୍ଚ ବିଦ୍ୟାଳୟ", font=font, fill='black')
        y_pos += 35
        draw.text((margin + 20, y_pos), "ଭୁବନେଶ୍ୱର", font=font, fill='black')
        y_pos += 80
        
        # Subject
        draw.text((margin, y_pos), "ବିଷୟ: ଅସୁସ୍ଥତା ହେତୁ ଛୁଟି ପାଇଁ ଆବେଦନ", font=font, fill='black')
        y_pos += 60
        
        # Salutation
        draw.text((margin, y_pos), "ମହୋଦୟ,", font=font, fill='black')
        y_pos += 50
        
        # Body
        body_text = """ସବିନୟ ନିବେଦନ ଯେ ମୁଁ ଆପଣଙ୍କ ବିଦ୍ୟାଳୟର ଦଶମ ଶ୍ରେଣୀର ଛାତ୍ର। ଗତକାଲିରୁ ମୋର ଜ୍ୱର ଓ ଶରୀର ଗୋଡ଼ାଣି ହେଉଛି।
        
ଡାକ୍ତରଙ୍କ ପରାମର୍ଶ ଅନୁସାରେ ମୋତେ ଦୁଇ ଦିନ ବିଶ୍ରାମ ନେବାକୁ ପଡ଼ିବ। ତେଣୁ ଆଜି ଓ ଆସନ୍ତାକାଲି ଦୁଇ ଦିନ ଛୁଟି ମଞ୍ଜୁର କରିବାକୁ ଅନୁରୋଧ।
        
ଆପଣଙ୍କ ବିଦ୍ୟାଳୟର ବାର୍ଷିକ ପରୀକ୍ଷା ନିକଟତର ହେଉଥିବାରୁ ମୁଁ ଶୀଘ୍ର ସୁସ୍ଥ ହୋଇ ପଢ଼ାଶୁଣାରେ ମନୋନିବେଶ କରିବି।"""
        
        paragraphs = body_text.strip().split('\n        \n')
        
        for paragraph in paragraphs:
            if paragraph.strip():
                lines = self.wrap_odia_text(paragraph.strip(), font, img_width - 2*margin)
                for line in lines:
                    draw.text((margin, y_pos), line, font=font, fill='black')
                    y_pos += 35
                y_pos += 20
        
        # Closing
        y_pos += 30
        draw.text((margin, y_pos), "ଧନ୍ୟବାଦ ସହ,", font=font, fill='black')
        y_pos += 80
        draw.text((img_width - 200, y_pos), "ଆପଣଙ୍କ ଆଜ୍ଞାକାରୀ ଛାତ୍ର", font=font, fill='black')
        y_pos += 40
        draw.text((img_width - 200, y_pos), "ରାମ କୁମାର ସାହୁ", font=font, fill='black')
        y_pos += 35
        draw.text((img_width - 200, y_pos), "ଦଶମ ଶ୍ରେଣୀ, 'କ' ବିଭାଗ", font=font, fill='black')
        
        # Save
        output_file = "odia_letter.png"
        img.save(output_file)
        print(f"Created letter format image: {output_file}")
        return output_file
    
    def wrap_odia_text(self, text, font, max_width):
        """Wrap Odia text to fit within specified width"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def create_all_formats(self):
        """Generate all different text format images"""
        print("Creating various Odia text image formats...")
        
        files_created = []
        
        try:
            files_created.append(self.create_paragraph_document())
        except Exception as e:
            print(f"Error creating document: {e}")
        
        try:
            files_created.append(self.create_poem_image())
        except Exception as e:
            print(f"Error creating poem: {e}")
        
        try:
            files_created.append(self.create_news_article())
        except Exception as e:
            print(f"Error creating news article: {e}")
        
        try:
            files_created.append(self.create_letter_format())
        except Exception as e:
            print(f"Error creating letter: {e}")
        
        return files_created

def main():
    generator = OdiaTextImageGenerator()
    created_files = generator.create_all_formats()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for file in created_files:
        if file:
            print(f"✓ Created: {file}")
    
    print(f"\nTotal files created: {len([f for f in created_files if f])}")
    print("All images are ready for use!")

if __name__ == "__main__":
    main()